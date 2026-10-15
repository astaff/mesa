/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "agx_nir_m5_mesh.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "util/u_math.h"
#include <stdlib.h>
#include <string.h>

bool
agx_nir_m5_canonicalize_empty_mesh(nir_shader *s)
{
   if (s->info.stage != MESA_SHADER_MESH ||
       (s->info.mesh.primitive_type != MESA_PRIM_POINTS &&
        s->info.mesh.primitive_type != MESA_PRIM_LINES) ||
       s->info.outputs_written || s->info.outputs_read)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_intrinsic_instr *count = NULL;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_set_vertex_and_primitive_count:
            if (count || nir_intrinsic_stream_id(intr))
               return false;
            for (unsigned i = 0; i < 2; ++i) {
               if (!nir_src_is_const(intr->src[i]) ||
                   nir_src_as_uint(intr->src[i]) != 0)
                  return false;
            }
            count = intr;
            break;
         case nir_intrinsic_store_output:
         case nir_intrinsic_store_per_vertex_output:
         case nir_intrinsic_store_per_primitive_output:
            return false;
         default:
            break;
         }
      }
   }
   if (!count)
      return false;

   nir_metadata_require(impl, nir_metadata_dominance);
   if (!nir_block_dominates(count->instr.block, nir_impl_last_block(impl)))
      return false;

   /* No geometry reaches rasterization. Keep the shader's memory work and
    * use the existing native triangle launch for the proven-empty output. */
   s->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   return true;
}

#define M5_MESH_MAX_PLANES (4 + 128 + 1 + 8)

struct mesh_layout {
   unsigned output_base, primitive_count, vertex_count, position;
   unsigned plane_stride, planes, bytes;
   bool storage_probe;
   const struct agx_m5_mesh_varyings *varyings;

   /* Per-primitive outputs (see agx_m5_mesh_unwelds): the API body writes
    * vertex planes to vertex_source and per-primitive planes to
    * primitive_source; the export region at position then holds three
    * private vertices per primitive. */
   bool unweld;
   unsigned export_vertices;
   unsigned vertex_source, vertex_stride;
   unsigned primitive_source, primitive_stride;
   int16_t layer_plane, viewport_plane;
   int16_t vertex_plane[M5_MESH_MAX_PLANES], primitive_plane[M5_MESH_MAX_PLANES];
};

static bool
agx_m5_mesh_unwelds(const struct agx_m5_mesh_varyings *varyings)
{
   if (varyings->lv)
      return true;
   for (unsigned slot = 0; slot < 64; ++slot) {
      if (varyings->primitive_comps[slot])
         return true;
   }
   return false;
}

static struct mesh_layout
mesh_layout(const nir_shader *s, const struct agx_m5_mesh_varyings *varyings)
{
   const unsigned base = ALIGN_POT(s->info.shared_size, 64);
   /* CPU-native maxV3/4/5/8/9/16/32/33/64/256 witnesses establish
    * a 16-byte-aligned SoA plane and aligned triangle-byte/count regions.
    * Native multi-varying witnesses (lane-mesh/native-var objects 9..17)
    * place one plane per UVS word directly after the four position planes,
    * in selector order. */
   const unsigned count = ALIGN_POT(3 * s->info.mesh.max_primitives_out, 16);
   const unsigned position = count + 16;
   const bool unweld = agx_m5_mesh_unwelds(varyings);
   const unsigned export_vertices = unweld
      ? 3 * s->info.mesh.max_primitives_out : s->info.mesh.max_vertices_out;
   const unsigned stride = ALIGN_POT(export_vertices, 4) * 4;
   struct mesh_layout layout = {
      .output_base = base, .primitive_count = base + count,
      .vertex_count = base + count + 4, .position = base + position,
      .plane_stride = stride, .planes = varyings->size,
      .bytes = ALIGN_POT(position + varyings->size * stride, 64),
      .storage_probe = getenv("ASAHI_MESH_PP_PROBE") != NULL &&
                       !strcmp(getenv("ASAHI_MESH_PP_PROBE"), "1"),
      .varyings = varyings,
      .unweld = unweld,
      .export_vertices = export_vertices,
      .layer_plane = -1, .viewport_plane = -1,
   };
   if (!unweld)
      return layout;

   /* Classify each exported UVS word as a vertex or primitive word. */
   unsigned vertex_planes = 0, primitive_planes = 0;
   for (unsigned plane = 0; plane < M5_MESH_MAX_PLANES; ++plane)
      layout.vertex_plane[plane] = layout.primitive_plane[plane] = -1;
   for (unsigned plane = 0; plane < MIN2(varyings->size, M5_MESH_MAX_PLANES); ++plane) {
      if (varyings->lv && plane == varyings->lv_first)
         continue; /* packed from the layer/viewport words */
      int primitive_slot = -1;
      for (unsigned slot = 0; slot < 64; ++slot) {
         if ((varyings->primitive_comps[slot] ||
              (varyings->lv && slot == VARYING_SLOT_LAYER)) &&
             plane >= varyings->first[slot] &&
             plane < varyings->first[slot] + varyings->components[slot])
            primitive_slot = slot;
      }
      if (primitive_slot < 0)
         layout.vertex_plane[plane] = vertex_planes++;
      else if (varyings->primitive_comps[primitive_slot])
         layout.primitive_plane[plane] = primitive_planes++;
      /* else: the unwritten layer word, exported as zero */
   }
   if (varyings->primitive_comps[VARYING_SLOT_LAYER])
      layout.layer_plane =
         layout.primitive_plane[varyings->first[VARYING_SLOT_LAYER]];
   if (varyings->primitive_comps[VARYING_SLOT_VIEWPORT])
      layout.viewport_plane =
         layout.primitive_plane[varyings->first[VARYING_SLOT_VIEWPORT]];

   layout.vertex_stride = ALIGN_POT(s->info.mesh.max_vertices_out, 4) * 4;
   layout.primitive_stride = ALIGN_POT(s->info.mesh.max_primitives_out, 4) * 4;
   const unsigned vertex_source = position + varyings->size * stride;
   const unsigned primitive_source =
      vertex_source + vertex_planes * layout.vertex_stride;
   layout.vertex_source = base + vertex_source;
   layout.primitive_source = base + primitive_source;
   layout.bytes = ALIGN_POT(primitive_source +
                            primitive_planes * layout.primitive_stride, 64);
   return layout;
}

/* Shared address of a vertex-output word written by the API body. */
static unsigned
vertex_plane_address(const struct mesh_layout *layout, unsigned plane)
{
   if (!layout->unweld)
      return layout->position + layout->plane_stride * plane;
   assert(layout->vertex_plane[plane] >= 0);
   return layout->vertex_source +
          layout->vertex_stride * layout->vertex_plane[plane];
}

static unsigned
vertex_output_slot(nir_intrinsic_instr *i)
{
   return nir_intrinsic_io_semantics(i).location + nir_src_as_uint(i->src[2]);
}

/* Plane (UVS word) of the first component of a per-vertex output slot. */
static unsigned
first_plane(const struct agx_m5_mesh_varyings *varyings, unsigned slot)
{
   return slot == VARYING_SLOT_POS ? 0 : varyings->first[slot];
}

static nir_def *
widen_output(nir_builder *b, nir_intrinsic_instr *i, nir_def *value)
{
   /* The UVS carries 32-bit words; FP16/INT16 varyings are converted the
    * same way agx_nir_lower_uvs does for the vertex stage, and the fragment
    * linkage (32-bit counts only) narrows them again on load. */
   if (value->bit_size == 32)
      return value;
   nir_alu_type type = nir_alu_type_get_base_type(nir_intrinsic_src_type(i));
   return nir_convert_to_bit_size(b, value, type, 32);
}

static bool
lower(nir_builder *b,nir_intrinsic_instr *i,void *data)
{
   const struct mesh_layout *layout = data;
   const unsigned output_base = layout->output_base;
   switch(i->intrinsic) {
   case nir_intrinsic_store_per_vertex_output: {
      b->cursor=nir_before_instr(&i->instr);
      nir_def *element=nir_imul_imm(b,i->src[1].ssa,4);
      const unsigned slot=vertex_output_slot(i);
      const unsigned first=first_plane(layout->varyings,slot);
      unsigned component=nir_intrinsic_component(i);
      nir_def *value=widen_output(b,i,i->src[0].ssa), *probe=NULL;
      if (slot == VARYING_SLOT_VAR0 && layout->storage_probe) {
         /* Opt-in read-only hardware probe (ASAHI_MESH_PP_PROBE=1): publish the
          * native per-primitive record base SR64/SR65 and the workgroup index
          * through the flat VAR0 planes instead of the API colour. No pointer
          * is dereferenced; the fragment dumps the raw bits as pixels. */
         nir_def *storage=nir_load_mesh_primitive_storage_agx(b);
         probe=nir_vec4(b,nir_channel(b,storage,0),nir_channel(b,storage,1),
                        nir_channel(b,nir_load_workgroup_id(b),0),
                        nir_load_local_invocation_index(b));
      }
      u_foreach_bit(c,nir_intrinsic_write_mask(i)) {
         nir_def *address=nir_iadd_imm(b,element,
            vertex_plane_address(layout,first+component+c));
         nir_store_shared(b,probe ? nir_channel(b,probe,component+c)
                                  : nir_channel(b,value,c),address,
                            .write_mask=1,.align_mul=4);
         if (slot == VARYING_SLOT_CLIP_DIST0 ||
             slot == VARYING_SLOT_CLIP_DIST1) {
            const unsigned clip =
               4 * (slot - VARYING_SLOT_CLIP_DIST0) + component + c;
            if (clip < layout->varyings->clip_distance_count) {
               nir_def *clip_address = nir_iadd_imm(b, element,
                  vertex_plane_address(layout,
                     layout->varyings->clip_distance_first + clip));
               nir_store_shared(b, nir_channel(b, value, c), clip_address,
                                .write_mask = 1, .align_mul = 4);
            }
         }
      }
      break;
   }
   case nir_intrinsic_store_per_primitive_output: {
      b->cursor=nir_before_instr(&i->instr);
      const unsigned slot = vertex_output_slot(i);
      if (slot != VARYING_SLOT_PRIMITIVE_INDICES) {
         /* A primitive word: the export below replicates it to the three
          * private vertices of its primitive. */
         nir_def *element = nir_imul_imm(b, i->src[1].ssa, 4);
         nir_def *value = widen_output(b, i, i->src[0].ssa);
         u_foreach_bit(c, nir_intrinsic_write_mask(i)) {
            int plane = layout->primitive_plane[layout->varyings->first[slot] +
                                                nir_intrinsic_component(i) + c];
            assert(plane >= 0);
            nir_store_shared(b, nir_channel(b, value, c),
               nir_iadd_imm(b, element, layout->primitive_source +
                                        layout->primitive_stride * plane),
               .write_mask = 1, .align_mul = 4);
         }
         break;
      }
      nir_def *element=nir_imul_imm(b,i->src[1].ssa,3);
      unsigned component=nir_intrinsic_component(i);
      u_foreach_bit(c,nir_intrinsic_write_mask(i)) {
         nir_def *index=nir_channel(b,i->src[0].ssa,c);
         /* Native helper creates per-byte validity masks with two ISELs,
          * adds low0xff/high0xff00, then ANDs the original index word.
          * Valid index bytes pass through unchanged. Cull is not mapped. */
         nir_def *packed=nir_u2u8(b,index);
         nir_store_shared(b,packed,nir_iadd_imm(b,element,output_base+component+c),
                            .write_mask=1,.align_mul=1);
      }
      break;
   }
   case nir_intrinsic_set_vertex_and_primitive_count: {
      b->cursor=nir_before_instr(&i->instr);
      nir_push_if(b,nir_ieq_imm(b,nir_load_local_invocation_index(b),0));
      nir_store_shared(b,nir_u2u16(b,i->src[1].ssa),nir_imm_int(b,layout->primitive_count),
                         .write_mask=1,.align_mul=2);
      nir_store_shared(b,nir_u2u16(b,i->src[0].ssa),nir_imm_int(b,layout->vertex_count),
                         .write_mask=1,.align_mul=2);
      nir_pop_if(b,NULL);
      break;
   }
   default:return false;
   }
   nir_instr_remove(&i->instr);
   return true;
}

#define USER_VARYINGS BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32)
#define MESH_VERTEX_VARYINGS \
   (USER_VARYINGS | VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1 | \
    BITFIELD64_RANGE(VARYING_SLOT_CULL_PRIMITIVE, 2))

bool
agx_m5_mesh_gather_varyings(const nir_shader *s,
                            struct agx_m5_mesh_varyings *varyings)
{
   memset(varyings, 0, sizeof(*varyings));
   if (s->info.clip_distance_array_size +
       s->info.cull_distance_array_size > 8)
      return false;
   varyings->clip_distance_count = s->info.clip_distance_array_size;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(in, block) {
            if (in->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
            if (i->intrinsic != nir_intrinsic_store_per_vertex_output &&
                i->intrinsic != nir_intrinsic_store_per_primitive_output)
               continue;
            if (!nir_src_is_const(i->src[2]))
               return false;
            const unsigned slot = vertex_output_slot(i);
            if (i->intrinsic == nir_intrinsic_store_per_primitive_output) {
               if (slot == VARYING_SLOT_PRIMITIVE_INDICES)
                  continue;
               /* Per-primitive user words become flat words of the
                * primitive's private vertices; layer/viewport fill the
                * vertex layer/viewport word. The fragment side reads both
                * as flat inputs (agx_nir_lower_m5_mesh_fragment). */
               const unsigned end = nir_intrinsic_component(i) +
                                    util_last_bit(nir_intrinsic_write_mask(i));
               const unsigned bits = i->src[0].ssa->bit_size;
               if (slot >= 64 || end > 4 || (bits != 32 && bits != 16))
                  return false;
               if (USER_VARYINGS & BITFIELD64_BIT(slot)) {
                  varyings->primitive_comps[slot] =
                     MAX2(varyings->primitive_comps[slot], end);
               } else if (slot == VARYING_SLOT_LAYER ||
                          slot == VARYING_SLOT_VIEWPORT) {
                  if (end != 1)
                     return false;
                  varyings->primitive_comps[slot] = 1;
                  varyings->lv = true;
               } else {
                  return false;
               }
               continue;
            }
            if (slot == VARYING_SLOT_POS)
               continue;
            if (slot >= 64 || !(MESH_VERTEX_VARYINGS & BITFIELD64_BIT(slot)))
               return false;
            const unsigned end = nir_intrinsic_component(i) +
                                 util_last_bit(nir_intrinsic_write_mask(i));
            if (end > 4)
               return false;
            varyings->components[slot] = MAX2(varyings->components[slot], end);
         }
      }
   }
   for (unsigned slot = VARYING_SLOT_VAR0; slot < 64; ++slot) {
      if (!varyings->primitive_comps[slot])
         continue;
      /* A location is either per-vertex or per-primitive. */
      if (varyings->components[slot])
         return false;
      varyings->components[slot] = varyings->primitive_comps[slot];
   }
   /* Like the M5 vertex stage (hk_lower_hw_vs with layer_as_varying), the
    * layer is also a flat user word for fragment reads, zero when only the
    * viewport is written, and a written viewport is one as well. */
   if (varyings->lv) {
      varyings->components[VARYING_SLOT_LAYER] = 1;
      varyings->components[VARYING_SLOT_VIEWPORT] =
         varyings->primitive_comps[VARYING_SLOT_VIEWPORT];
   }
   /* Declaration order. A single user slot (or one interpolation class) sits
    * at the base of the user section for every fragment shader; otherwise the
    * fragment shader chooses the class order at draw time. The UVS groups
    * that follow are layer/viewport, then clip distances. */
   unsigned next = 4, slots = 0;
   for (unsigned slot = 0; slot < 64; ++slot) {
      if (!varyings->components[slot])
         continue;
      varyings->first[slot] = next;
      next += varyings->components[slot];
      slots++;
   }
   if (varyings->lv)
      varyings->lv_first = next++;
   varyings->clip_distance_first = next;
   varyings->size = next + varyings->clip_distance_count;
   varyings->dynamic = slots > 1;
   return true;
}

/* Every user plane is covered exactly once by the slot placement. */
static bool
varyings_valid(const struct agx_m5_mesh_varyings *v)
{
   const unsigned lv = v->lv ? 1 : 0;
   if (v->size < 4 || v->size > 4 + 128 + 8 + lv ||
       v->clip_distance_count > 8 ||
       v->size < 4 + lv + v->clip_distance_count ||
       (v->clip_distance_count &&
        v->clip_distance_first + v->clip_distance_count != v->size))
      return false;
   /* User words, then the optional layer/viewport word, then clip words. */
   const unsigned user_end = v->size - v->clip_distance_count - lv;
   if (v->lv && v->lv_first != user_end)
      return false;
   uint8_t used[M5_MESH_MAX_PLANES] = {0};
   unsigned user = 0;
   for (unsigned slot = 0; slot < 64; ++slot) {
      if (!v->components[slot])
         continue;
      const bool lv_word = v->lv && (slot == VARYING_SLOT_LAYER ||
                                     slot == VARYING_SLOT_VIEWPORT);
      if (lv_word) {
         if (v->components[slot] != 1 ||
             (slot == VARYING_SLOT_VIEWPORT && !v->primitive_comps[slot]))
            return false;
      } else if (!(MESH_VERTEX_VARYINGS & BITFIELD64_BIT(slot)) ||
                 (v->primitive_comps[slot] &&
                  (!(USER_VARYINGS & BITFIELD64_BIT(slot)) ||
                   v->primitive_comps[slot] != v->components[slot]))) {
         return false;
      }
      if (v->components[slot] > 4 ||
          v->first[slot] < 4 ||
          v->first[slot] + v->components[slot] > user_end)
         return false;
      for (unsigned c = 0; c < v->components[slot]; ++c) {
         if (used[v->first[slot] + c]++)
            return false;
      }
      user += v->components[slot];
   }
   if (v->lv != (v->primitive_comps[VARYING_SLOT_LAYER] ||
                 v->primitive_comps[VARYING_SLOT_VIEWPORT]) ||
       (v->lv && !v->components[VARYING_SLOT_LAYER]))
      return false;
   return user + 4 + lv + v->clip_distance_count == v->size;
}

bool
agx_nir_lower_m5_mesh_triangle_probe(nir_shader *s)
{
   struct agx_m5_mesh_varyings varyings;
   if (!agx_m5_mesh_gather_varyings(s, &varyings))
      return false;
   return agx_nir_lower_m5_mesh(s, &varyings);
}

bool
agx_nir_lower_m5_mesh(nir_shader *s, const struct agx_m5_mesh_varyings *varyings)
{
   const uint64_t primitive_outputs =
      s->info.per_primitive_outputs & s->info.outputs_written;
   uint64_t primitive_slots = VARYING_BIT_PRIMITIVE_INDICES;
   for (unsigned slot = 0; slot < 64; ++slot) {
      if (varyings->primitive_comps[slot])
         primitive_slots |= BITFIELD64_BIT(slot);
   }
   const bool unweld = agx_m5_mesh_unwelds(varyings);
   unsigned threads=s->info.workgroup_size[0]*s->info.workgroup_size[1]*
                    s->info.workgroup_size[2];
   if(s->info.stage!=MESA_SHADER_MESH || s->info.shared_size>32768 ||
      s->info.workgroup_size_variable || threads==0 || threads>1024 ||
      s->info.mesh.max_vertices_out==0 || s->info.mesh.max_vertices_out>256 ||
      s->info.mesh.max_primitives_out==0 || s->info.mesh.max_primitives_out>256 ||
      s->info.mesh.primitive_type!=MESA_PRIM_TRIANGLES ||
      /* Interstage compaction can leave per_primitive_outputs stale; with
       * per-primitive words every store is checked against them below. */
      (primitive_outputs&~(unweld ? ~UINT64_C(0) : primitive_slots)) ||
      s->info.outputs_written & ~(VARYING_BIT_POS|VARYING_BIT_PRIMITIVE_INDICES|MESH_VERTEX_VARYINGS|
                                  (primitive_slots & (VARYING_BIT_LAYER|VARYING_BIT_VIEWPORT))) ||
      /* Every primitive gets three private vertices within the native
       * 256-vertex output limit. */
      (unweld && 3 * s->info.mesh.max_primitives_out > 256) ||
      varyings->clip_distance_count != s->info.clip_distance_array_size ||
      !varyings_valid(varyings))
      return false;
   struct mesh_layout layout = mesh_layout(s, varyings);
   if (layout.output_base + layout.bytes > 32768)
      return false;
   unsigned physical_threads=MAX2(threads,MIN2(layout.export_vertices,32));
   bool counts=false;
   nir_foreach_function_impl(impl,s) {
      nir_foreach_block(block,impl) {
         nir_foreach_instr(in,block) {
            if(in->type!=nir_instr_type_intrinsic)continue;
            nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
            switch(i->intrinsic) {
            case nir_intrinsic_set_vertex_and_primitive_count:counts=true;break;
            case nir_intrinsic_barrier:
               /* Padding moves the API body under its exact active mask.
                * A user execution barrier needs a separate reconvergence
                * lowering; do not silently change its participating set. */
               if (physical_threads != threads &&
                   nir_intrinsic_execution_scope(i) != SCOPE_NONE)
                  return false;
               break;
            case nir_intrinsic_store_per_vertex_output: {
               if (!nir_src_is_const(i->src[2]))
                  return false;
               const unsigned slot=vertex_output_slot(i);
               const unsigned bits=i->src[0].ssa->bit_size;
               const unsigned end=nir_intrinsic_component(i)+
                                  util_last_bit(nir_intrinsic_write_mask(i));
               if(slot>=64 || (bits!=32 && bits!=16) || end>4)
                  return false;
               if(slot==VARYING_SLOT_POS) {
                  if(bits!=32)return false;
               } else if(!varyings->components[slot] ||
                         end>varyings->components[slot]) {
                  return false;
               }
               break;
            }
            case nir_intrinsic_store_per_primitive_output: {
               if (!nir_src_is_const(i->src[2]))
                  return false;
               const unsigned slot=vertex_output_slot(i);
               const unsigned bits=i->src[0].ssa->bit_size;
               const unsigned end=nir_intrinsic_component(i)+
                                  util_last_bit(nir_intrinsic_write_mask(i));
               if (slot==VARYING_SLOT_PRIMITIVE_INDICES) {
                  if(nir_intrinsic_io_semantics(i).location!=VARYING_SLOT_PRIMITIVE_INDICES ||
                     nir_intrinsic_component(i)+i->src[0].ssa->num_components>3 ||
                     nir_src_as_uint(i->src[2])!=0)
                     return false;
               } else if (slot>=64 || !varyings->primitive_comps[slot] ||
                          end>varyings->primitive_comps[slot] ||
                          (bits!=32 && bits!=16)) {
                  return false;
               }
               break;
            }
            case nir_intrinsic_load_per_vertex_output:
            case nir_intrinsic_load_per_primitive_output:
            case nir_intrinsic_launch_mesh_workgroups:return false;
            default:break;
            }
         }
      }
   }
   if (!counts) {
      if (s->info.outputs_written || s->info.outputs_read)
         return false;
      nir_builder b = nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(s)));
      nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 0),
         nir_imm_int(&b, 0), nir_imm_int(&b, 0));
   }
   if (physical_threads != threads || s->info.workgroup_size[2] > 1023) {
      /* Native output batches target vertex_base + physical lane. Pad at
       * most the first output SIMD group; later batches reuse it with the
       * captured high16 vertex_base field. Keep the original API body mask. */
      nir_function_impl *entry=nir_shader_get_entrypoint(s);
      nir_foreach_block(block,entry) {
         nir_foreach_instr_safe(in,block) {
            if (in->type != nir_instr_type_intrinsic) continue;
            nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
            nir_builder b=nir_builder_at(nir_before_instr(in));
            nir_def *replacement=NULL;
            if (i->intrinsic == nir_intrinsic_load_workgroup_size) {
               replacement=nir_imm_ivec3(&b,s->info.workgroup_size[0],
                  s->info.workgroup_size[1],s->info.workgroup_size[2]);
            } else if (i->intrinsic == nir_intrinsic_load_local_invocation_id ||
                       i->intrinsic == nir_intrinsic_load_global_invocation_id) {
               nir_def *index=nir_load_local_invocation_index(&b);
               unsigned x=s->info.workgroup_size[0], y=s->info.workgroup_size[1];
               replacement=nir_vec3(&b,nir_umod_imm(&b,index,x),
                  nir_umod_imm(&b,nir_udiv_imm(&b,index,x),y),
                  nir_udiv_imm(&b,index,x*y));
               if (i->intrinsic == nir_intrinsic_load_global_invocation_id) {
                  /* The hardware thread position in grid counts the padded
                   * physical workgroup; the API id uses the declared size. */
                  nir_def *size=nir_imm_ivec3(&b,s->info.workgroup_size[0],
                     s->info.workgroup_size[1],s->info.workgroup_size[2]);
                  replacement=nir_iadd(&b,nir_imul(&b,nir_load_workgroup_id(&b),size),
                                       replacement);
                  replacement=nir_u2uN(&b,replacement,i->def.bit_size);
               }
            }
            if (replacement) {
               nir_def_rewrite_uses(&i->def,replacement);
               nir_instr_remove(in);
            }
         }
      }
      nir_cf_list body;
      nir_cf_extract(&body,nir_before_cf_list(&entry->body),
                            nir_after_cf_list(&entry->body));
      nir_builder b=nir_builder_at(nir_before_impl(entry));
      nir_if *active=nir_push_if(&b,nir_ult_imm(&b,
                                  nir_load_local_invocation_index(&b),threads));
      b.cursor=nir_cf_reinsert(&body,b.cursor);
      nir_pop_if(&b,active);
      s->info.workgroup_size[0]=physical_threads;
      s->info.workgroup_size[1]=s->info.workgroup_size[2]=1;
   }
   /* Keep application shared storage disjoint from the native output arena.
    * The UVST descriptor carries a byte address, so relocation changes no
    * primitive/position plane layout. Existing barriers preserve visibility. */
   nir_shader_intrinsics_pass(s,lower,nir_metadata_none,&layout);
   nir_function_impl *impl=nir_shader_get_entrypoint(s);
   nir_builder b=nir_builder_at(nir_after_cf_list(&impl->body));
   const unsigned max_primitives = s->info.mesh.max_primitives_out;
   if (layout.unweld) {
      /* Per-primitive words have no native vertex form here. Give every
       * primitive three private vertices: copy each referenced vertex's
       * words, replicate the primitive's words (flat on every corner, so
       * any provoking vertex selects them), and rewrite the index list to
       * the identity. Primitive order, winding and the interpolated values
       * are unchanged. SIMD group 0 copies, in the same 32-lane batches
       * as the export, between two rendezvous: UVB stores read only words
       * published by a threadgroup barrier, as for API output stores (a
       * SIMD-group barrier between the copy and the export left stale
       * words on hardware). Each lane reads and then rewrites only the
       * index bytes of its own private vertices. */
      nir_barrier(&b,.execution_scope=SCOPE_WORKGROUP,.memory_scope=SCOPE_WORKGROUP,
                     .memory_semantics=NIR_MEMORY_ACQ_REL,.memory_modes=nir_var_mem_shared);
      nir_push_if(&b, nir_ieq_imm(&b, nir_load_subgroup_id(&b), 0));
      nir_def *primitives = nir_umin(&b, nir_u2u32(&b, nir_load_shared(&b, 1, 16,
         nir_imm_int(&b, layout.primitive_count), .align_mul = 2)),
         nir_imm_int(&b, max_primitives));
      nir_def *vertices = nir_imul_imm(&b, primitives, 3);
      nir_def *lane = nir_load_subgroup_invocation(&b);
      for (unsigned base = 0; base < 3 * max_primitives; base += 32) {
         nir_def *j = nir_iadd_imm(&b, lane, base);
         nir_push_if(&b, nir_ult(&b, j, vertices));
         nir_def *index = nir_u2u32(&b, nir_load_shared(&b, 1, 8,
            nir_iadd_imm(&b, j, layout.output_base), .align_mul = 1));
         nir_def *source = nir_imul_imm(&b,
            nir_umin(&b, index, nir_imm_int(&b, s->info.mesh.max_vertices_out - 1)), 4);
         nir_def *primitive = nir_imul_imm(&b, nir_udiv_imm(&b, j, 3), 4);
         nir_def *target = nir_imul_imm(&b, j, 4);
         for (unsigned plane = 0; plane < layout.planes; ++plane) {
            nir_def *value;
            if (varyings->lv && plane == varyings->lv_first) {
               nir_def *layer = nir_imm_int(&b, 0), *viewport = nir_imm_int(&b, 0);
               if (layout.layer_plane >= 0)
                  layer = nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, primitive,
                     layout.primitive_source + layout.primitive_stride * layout.layer_plane),
                     .align_mul = 4);
               if (layout.viewport_plane >= 0)
                  viewport = nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, primitive,
                     layout.primitive_source + layout.primitive_stride * layout.viewport_plane),
                     .align_mul = 4);
               /* render_target_array_index low, viewport_array_index high,
                * as agx_nir_lower_uvs packs the vertex-stage word. */
               value = nir_pack_32_2x16_split(&b, nir_u2u16(&b, layer),
                                              nir_u2u16(&b, viewport));
            } else if (layout.vertex_plane[plane] >= 0) {
               value = nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, source,
                  layout.vertex_source + layout.vertex_stride * layout.vertex_plane[plane]),
                  .align_mul = 4);
            } else if (layout.primitive_plane[plane] < 0) {
               value = nir_imm_int(&b, 0);
            } else {
               value = nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, primitive,
                  layout.primitive_source +
                  layout.primitive_stride * layout.primitive_plane[plane]),
                  .align_mul = 4);
            }
            nir_store_shared(&b, value, nir_iadd_imm(&b, target,
               layout.position + layout.plane_stride * plane), .write_mask = 1,
               .align_mul = 4);
         }
         nir_store_shared(&b, nir_u2u8(&b, j),
            nir_iadd_imm(&b, j, layout.output_base), .write_mask = 1,
            .align_mul = 1);
         nir_pop_if(&b, NULL);
      }
      nir_pop_if(&b, NULL);
   }
   nir_barrier(&b,.execution_scope=SCOPE_WORKGROUP,.memory_scope=SCOPE_WORKGROUP,
                  .memory_semantics=NIR_MEMORY_ACQ_REL,.memory_modes=nir_var_mem_shared);
   /* Native write_uvb helpers execute on SIMD group0 after rendezvous. */
   nir_push_if(&b,nir_ieq_imm(&b,nir_load_subgroup_id(&b),0));
   nir_def *p=nir_load_shared(&b,1,16,nir_imm_int(&b,layout.primitive_count),.align_mul=2);
   nir_def *v=nir_load_shared(&b,1,16,nir_imm_int(&b,layout.vertex_count),.align_mul=2);
   if (layout.unweld) {
      p = nir_umin(&b, p, nir_imm_intN_t(&b, max_primitives, 16));
      v = nir_imul_imm(&b, p, 3);
   }
   nir_allocate_mesh_uvb_agx(&b,p,v);
   nir_def *lane=nir_load_subgroup_invocation(&b);
   nir_def *primitive_words=nir_udiv_imm(&b,
      nir_iadd_imm(&b,nir_imul_imm(&b,nir_u2u32(&b,p),3),3),4);
   for (unsigned word=0;word<DIV_ROUND_UP(3*s->info.mesh.max_primitives_out,4);word+=32) {
      nir_push_if(&b,nir_ult(&b,nir_iadd_imm(&b,lane,word),primitive_words));
      /* Native64/256 helper increments high16 target word by32 and
       * shared-byte address by128, while each active lane emits one word. */
      nir_store_mesh_uvb_agx(&b,nir_imm_ivec2(&b,word<<16,
         layout.output_base+4*word),.flags=1);
      nir_pop_if(&b,NULL);
   }
   /* Plane k publishes UVS word k unless the fragment interpolation order is
    * a draw-time choice; then user planes take the root UVS index of their
    * slot, like an unlinked vertex shader's store_uvs_agx. */
   nir_def *selector[M5_MESH_MAX_PLANES] = {NULL};
   if (varyings->dynamic) {
      for (unsigned slot = 0; slot < 64; ++slot) {
         if (!varyings->components[slot])
            continue;
         nir_def *base = nir_u2u32(&b, nir_load_uvs_index_agx(&b,
            .io_semantics = {.location = slot, .num_slots = 1}));
         for (unsigned c = 0; c < varyings->components[slot]; ++c)
            selector[varyings->first[slot] + c] = nir_iadd_imm(&b, base, c);
      }
   }
   for(unsigned vertex=0;vertex<layout.export_vertices;vertex+=32) {
      nir_push_if(&b,nir_ult(&b,nir_iadd_imm(&b,lane,vertex),nir_u2u32(&b,v)));
      for(unsigned plane=0;plane<layout.planes;plane++) {
         const unsigned address=layout.position+layout.plane_stride*plane+4*vertex;
         /* Native mesh64 stores0x200000|component in round2. The
          * descriptor shared address names the batch base, as native does. */
         nir_def *descriptor = selector[plane]
            ? nir_vec2(&b, nir_ior_imm(&b, selector[plane], vertex << 16),
                       nir_imm_int(&b, address))
            : nir_imm_ivec2(&b, (vertex << 16) | plane, address);
         nir_store_mesh_uvb_agx(&b,descriptor,.flags=0);
      }
      nir_pop_if(&b,NULL);
   }
   nir_pop_if(&b,NULL);
   s->info.shared_size=layout.output_base+layout.bytes;
   s->info.outputs_written=0;s->info.per_primitive_outputs=0;
   /* The native output limits describe the exported vertices. */
   s->info.mesh.max_vertices_out=layout.export_vertices;
   nir_progress(true,impl,nir_metadata_none);
   return true;
}

/* agx_nir_lower_m5_mesh gives every primitive private vertices and stores
 * its per-primitive user, layer and viewport words on all of them, so the
 * fragment shader reads them as ordinary flat inputs. Like the mesh side,
 * UVS words are 32-bit; narrower inputs are narrowed after the load. */
static bool
lower_m5_primitive_input(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   (void)data;
   if (i->intrinsic != nir_intrinsic_load_per_primitive_input)
      return false;
   nir_io_semantics sem = nir_intrinsic_io_semantics(i);
   const unsigned location = sem.location;
   if (!((location >= VARYING_SLOT_VAR0 && location < VARYING_SLOT_VAR0 + 32) ||
         location == VARYING_SLOT_LAYER || location == VARYING_SLOT_VIEWPORT) ||
       (i->def.bit_size != 32 && i->def.bit_size != 16))
      return false;
   b->cursor = nir_before_instr(&i->instr);
   const nir_alu_type base =
      nir_alu_type_get_base_type(nir_intrinsic_dest_type(i));
   nir_def *value = nir_load_input(b, i->def.num_components, 32,
      i->src[0].ssa, .base = nir_intrinsic_base(i),
      .component = nir_intrinsic_component(i), .dest_type = base | 32,
      .io_semantics = sem);
   if (i->def.bit_size != 32)
      value = nir_convert_to_bit_size(b, value, base, i->def.bit_size);
   b->shader->info.per_primitive_inputs &=
      ~BITFIELD64_RANGE(location, MAX2(sem.num_slots, 1));
   nir_def_replace(&i->def, value);
   return true;
}

bool
agx_nir_lower_m5_mesh_fragment(nir_shader *s)
{
   if (s->info.stage != MESA_SHADER_FRAGMENT)
      return false;
   return nir_shader_intrinsics_pass(s, lower_m5_primitive_input,
                                     nir_metadata_control_flow, NULL);
}

/* The retained native object and matching mesh both access the same pointer:
 * normalized SR62/W32 + SR63/W16. The four-byte payload limit is the actual
 * captured source contract, not a claimed hardware maximum. */
static bool
lower_payload(nir_builder *b,nir_intrinsic_instr *i,void *data)
{
   (void)data;
   if (i->intrinsic==nir_intrinsic_barrier &&
       (nir_intrinsic_memory_modes(i)&nir_var_mem_task_payload)) {
      nir_intrinsic_set_memory_modes(i,
         (nir_intrinsic_memory_modes(i)&~nir_var_mem_task_payload)|nir_var_mem_global);
      return true;
   }
   bool store=i->intrinsic==nir_intrinsic_store_task_payload;
   if (!store && i->intrinsic!=nir_intrinsic_load_task_payload) return false;
   b->cursor=nir_before_instr(&i->instr);
   nir_def *pointer=nir_pack_64_2x32(b,nir_load_mesh_payload_address_agx(b));
   nir_def *offset=i->src[store ? 1 : 0].ssa;
   nir_def *addr=nir_iadd(b,pointer,nir_u2u64(b,
                            nir_iadd_imm(b,offset,nir_intrinsic_base(i))));
   if (store) {
      nir_store_global(b,i->src[0].ssa,addr,
         .write_mask=nir_intrinsic_write_mask(i),
         .align_mul=nir_intrinsic_align_mul(i),
         .align_offset=nir_intrinsic_align_offset(i),
         .access=nir_intrinsic_access(i));
      nir_instr_remove(&i->instr);
   } else {
      nir_def_replace(&i->def,nir_load_global(b,i->def.num_components,
         i->def.bit_size,addr,.align_mul=nir_intrinsic_align_mul(i),
         .align_offset=nir_intrinsic_align_offset(i),
         .access=nir_intrinsic_access(i)));
   }
   return true;
}

bool
agx_nir_lower_m5_task_payload_probe(nir_shader *s)
{
   if (s->info.task_payload_size>4) return false;
   bool reads=false;
   nir_foreach_function_impl(impl,s) {
      nir_foreach_block(block,impl) {
         nir_foreach_instr(in,block) {
            if (in->type!=nir_instr_type_intrinsic) continue;
            nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
            reads |= i->intrinsic==nir_intrinsic_load_task_payload;
            if (i->intrinsic==nir_intrinsic_task_payload_atomic ||
                i->intrinsic==nir_intrinsic_task_payload_atomic_swap ||
                i->intrinsic==nir_intrinsic_launch_mesh_workgroups_with_payload_deref)
               return false;
         }
      }
   }
   if (reads && s->info.stage==MESA_SHADER_MESH) {
      nir_builder b=nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(s)));
      nir_mesh_payload_entry_barrier_agx(&b);
   }
   nir_shader_intrinsics_pass(s,lower_payload,nir_metadata_control_flow,NULL);
   return true;
}

static bool
lower_launch(nir_builder *b,nir_intrinsic_instr *i,void *data)
{
   if (i->intrinsic!=nir_intrinsic_launch_mesh_workgroups) return false;
   const unsigned grid=*(unsigned *)data;
   b->cursor=nir_before_instr(&i->instr);
   nir_push_if(b,nir_ieq_imm(b,nir_load_local_invocation_index(b),0));
   nir_store_shared(b,i->src[0].ssa,nir_imm_int(b,grid),
                       .write_mask=7,.align_mul=4);
   nir_pop_if(b,NULL);
   nir_instr_remove(&i->instr);
   return true;
}

bool
agx_nir_lower_m5_task_probe(nir_shader *s)
{
   unsigned threads=s->info.workgroup_size[0]*s->info.workgroup_size[1]*
                    s->info.workgroup_size[2];
   if (s->info.stage!=MESA_SHADER_TASK || s->info.workgroup_size_variable ||
       threads<3 || threads>128 || s->info.shared_size>48 ||
       !agx_nir_lower_m5_task_payload_probe(s)) return false;
   unsigned launches=0;
   nir_foreach_function_impl(impl,s) {
      nir_foreach_block(block,impl) {
         nir_foreach_instr(in,block) {
            if (in->type==nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(in)->intrinsic==nir_intrinsic_launch_mesh_workgroups)
               launches++;
         }
      }
   }
   if (!launches) return false;
   unsigned grid=ALIGN_POT(s->info.shared_size,16);
   nir_shader_intrinsics_pass(s,lower_launch,nir_metadata_none,&grid);
   nir_builder b=nir_builder_at(nir_after_cf_list(&nir_shader_get_entrypoint(s)->body));
   nir_barrier(&b,.execution_scope=SCOPE_WORKGROUP,.memory_scope=SCOPE_WORKGROUP,
      .memory_semantics=NIR_MEMORY_ACQ_REL,
      .memory_modes=nir_var_mem_shared);
   nir_object_payload_exit_barrier_agx(&b);
   nir_def *lane=nir_load_local_invocation_index(&b);
   nir_push_if(&b,nir_ult_imm(&b,lane,3));
   nir_def *value=nir_load_shared(&b,1,32,
      nir_iadd_imm(&b,nir_imul_imm(&b,lane,4),grid),.align_mul=4);
   nir_store_object_childcount_agx(&b,value);
   nir_pop_if(&b,NULL);
   s->info.shared_size=64;
   nir_progress(true,b.impl,nir_metadata_none);
   return true;
}
