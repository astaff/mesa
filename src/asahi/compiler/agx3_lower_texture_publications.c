/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx3_staging.h"
#include <string.h>

struct publication {
   agx_index coordinates, reference;
   uint32_t value;
   unsigned cost;
};

static bool
same_register_value(agx_index a, agx_index b)
{
   /* Cache/discard annotations describe individual reads, not values. */
   return a.type == b.type && a.value == b.value && a.size == b.size &&
          a.channels_m1 == b.channels_m1 && a.memory == b.memory &&
          a.abs == b.abs && a.neg == b.neg;
}

static bool
overlaps_register(agx_index a, agx_index b)
{
   return a.type == AGX_INDEX_REGISTER && b.type == AGX_INDEX_REGISTER &&
          a.memory == b.memory &&
          a.value < b.value + agx_index_size_16(b) &&
          b.value < a.value + agx_index_size_16(a);
}

/* Native register-only instructions preserve the sampler-private file. Keep
 * compound memory/texture operations, control flow, coverage changes and
 * arithmetic publications as region boundaries until their native resource
 * effects are represented. This classifies effects; it does not reorder them.
 */
static bool
preserves_publications(const agx_instr *I)
{
   if (agx_texture_resource(I))
      return true;
   const struct agx_opcode_info *info = &agx_opcodes_info[I->op];
   if (I->agx3_tex_coord_stage || I->nr_dests != 1 ||
       I->dest[0].type != AGX_INDEX_REGISTER || I->dest[0].memory ||
       !info->can_eliminate || !info->can_reorder ||
       info->schedule_class != AGX_SCHEDULE_CLASS_NONE)
      return false;
   switch (I->op) {
   case AGX_OPCODE_TEXTURE_SAMPLE:
   case AGX_OPCODE_TEXTURE_LOAD:
   case AGX_OPCODE_IMAGE_LOAD:
   case AGX_OPCODE_INTERP_AGX3:
   case AGX_OPCODE_ITRCOORDS_AGX3:
   case AGX_OPCODE_RHW_INTERP_AGX3:
   case AGX_OPCODE_LDCF:
      return false;
   default:
      return true;
   }
}

/* This is the relocatable coordinate ABI, independent of shader identity or
 * instruction neighbourhood. Other native operand ABIs still have compound
 * lowering. Arithmetic-to-staging retains its helper-lane provenance.
 */
static bool
has_coordinate_publication(const agx_instr *I)
{
   if (I->op != AGX_OPCODE_TEXTURE_SAMPLE || agx_texture_resource(I) ||
       I->agx3_tex_coords_staged || I->dim != AGX_DIM_2D || I->query_lod ||
       (I->offset && !I->offset_xyz) || I->nr_srcs != 6 ||
       I->src[0].type != AGX_INDEX_REGISTER || I->src[0].memory ||
       I->src[0].abs || I->src[0].neg || I->src[0].size != AGX_SIZE_32 ||
       agx_channels(I->src[0]) != 2)
      return false;

   if (I->shadow && (I->src[5].type != AGX_INDEX_REGISTER ||
                    I->src[5].memory || I->src[5].abs || I->src[5].neg ||
                    I->src[5].size != AGX_SIZE_32))
      return false;

   return I->lod_mode == AGX_LOD_MODE_AUTO_LOD ||
          ((I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS ||
            I->lod_mode == AGX_LOD_MODE_LOD_MIN) &&
           I->src[1].type == AGX_INDEX_IMMEDIATE);
}

bool
agx3_lower_texture_publications(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return false;

   uint32_t next = 0;
   agx_foreach_instr_global(ctx, I)
      next = MAX3(next, agx_texture_resource(I), agx_publication_resource(I));

   bool progress = false;
   agx_foreach_block(ctx, block) {
      /* At most one four-half pair fits per entry. Reserving each available
       * value's aligned footprint bounds future-use overlap by the actual
       * register-file capacity, independently of the number of samples.
       * Eviction ends reuse eligibility; it never removes a required copy.
       */
      struct publication available[AGX3_STAGING_HALFS / 4];
      unsigned count = 0, reserved = 0;
      agx_foreach_instr_in_block_safe(block, I) {
         if (has_coordinate_publication(I)) {
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            agx_index reference = agx_null();
            if (I->shadow) {
               reference = I->src[5];
               reference.channels_m1 = 0;
            }
            unsigned match = count;
            for (unsigned j = 0; j < count; ++j) {
               if (same_register_value(available[j].coordinates, I->src[0]) &&
                   same_register_value(available[j].reference, reference)) {
                  match = j;
                  break;
               }
            }
            uint32_t value;
            if (match < count) {
               const struct publication hit = available[match];
               value = hit.value;
               memmove(&available[match], &available[match + 1],
                        (count - match - 1) * sizeof(*available));
               available[count - 1] = hit;
            } else {
               const unsigned cost = I->shadow ? 8 : 4;
               while (reserved + cost > AGX3_STAGING_HALFS) {
                  reserved -= available[0].cost;
                  memmove(available, available + 1, --count * sizeof(*available));
               }
               agx_publish_texture_agx3(&b, I->src[0], reference, ++next);
               value = next;
               available[count++] = (struct publication){
                  I->src[0], reference, value, cost,
               };
               reserved += cost;
            }
            /* The resource, not its original GPRs, feeds later samples. */
            I->src[0] = agx_get_vec_index(value, AGX_SIZE_32, I->shadow ? 3 : 2);
            I->src[0].type = AGX_INDEX_RESOURCE;
            if (I->shadow)
               I->src[5] = agx_null();
            progress = true;
         }

         if (!preserves_publications(I)) {
            count = reserved = 0;
            continue;
         }
         /* Any overlapping physical definition changes the input value,
          * including an asynchronous sample result or a half-register write.
          */
         for (unsigned j = 0; j < count;) {
            bool overwritten = false;
            agx_foreach_dest(I, d) {
               overwritten |= overlaps_register(I->dest[d], available[j].coordinates) ||
                              overlaps_register(I->dest[d], available[j].reference);
            }
            if (!overwritten) {
               ++j;
               continue;
            }
            reserved -= available[j].cost;
            memmove(&available[j], &available[j + 1],
                     (--count - j) * sizeof(*available));
         }
      }
   }
   return progress;
}

static bool
scalar_publication_source(agx_index src)
{
   return (src.type == AGX_INDEX_NORMAL || src.type == AGX_INDEX_REGISTER) &&
          !src.memory && !src.abs && !src.neg && src.size == AGX_SIZE_32 &&
          agx_channels(src) == 1;
}

static bool
ssa_preserves_publications(const agx_instr *I)
{
   if (agx_texture_resource(I))
      return true;
   const struct agx_opcode_info *info = &agx_opcodes_info[I->op];
   if (I->agx3_tex_coord_stage || !I->nr_dests ||
       !info->can_eliminate || !info->can_reorder ||
       info->schedule_class != AGX_SCHEDULE_CLASS_NONE)
      return false;
   /* SPLIT/collect are SSA wiring, not native resource effects. In particular,
    * splitting a texture result must not discard its still-live coordinates.
    */
   agx_foreach_dest(I, d) {
      if ((I->dest[d].type != AGX_INDEX_NORMAL &&
           I->dest[d].type != AGX_INDEX_NULL) || I->dest[d].memory)
         return false;
   }
   switch (I->op) {
   case AGX_OPCODE_TEXTURE_SAMPLE:
   case AGX_OPCODE_TEXTURE_LOAD:
   case AGX_OPCODE_IMAGE_LOAD:
   case AGX_OPCODE_INTERP_AGX3:
   case AGX_OPCODE_ITRCOORDS_AGX3:
   case AGX_OPCODE_RHW_INTERP_AGX3:
   case AGX_OPCODE_LDCF:
      return false;
   default:
      return true;
   }
}

bool
agx3_lower_scalar_texture_publications(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble)
      return false;

   agx_instr **defs = rzalloc_array(ctx, agx_instr *, ctx->alloc);
   if (!defs)
      return false;
   uint32_t next = 0;
   agx_foreach_instr_global(ctx, I) {
      next = MAX3(next, agx_texture_resource(I), agx_publication_resource(I));
      agx_foreach_ssa_dest(I, d)
         defs[I->dest[d].value] = I;
   }

   struct scalar_publication {
      agx_index x, y, reference;
      uint32_t value;
      unsigned cost;
   };
   bool progress = false;
   agx_foreach_block(ctx, block) {
      struct scalar_publication available[AGX3_STAGING_HALFS / 4];
      unsigned count = 0, reserved = 0;
      agx_foreach_instr_in_block_safe(block, I) {
         bool eligible =
            I->op == AGX_OPCODE_TEXTURE_SAMPLE && !agx_texture_resource(I) &&
            !I->agx3_tex_coords_staged && I->dim == AGX_DIM_2D && !I->query_lod &&
            (!I->offset || I->offset_xyz) && I->nr_srcs == 6 &&
            I->src[0].type == AGX_INDEX_NORMAL && !I->src[0].memory &&
            !I->src[0].abs && !I->src[0].neg && I->src[0].size == AGX_SIZE_32 &&
            agx_channels(I->src[0]) == 2 &&
            (I->lod_mode == AGX_LOD_MODE_AUTO_LOD ||
             ((I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS ||
               I->lod_mode == AGX_LOD_MODE_LOD_MIN) &&
              I->src[1].type == AGX_INDEX_IMMEDIATE));
         agx_instr *collect = eligible ? defs[I->src[0].value] : NULL;
         eligible &= collect && collect->op == AGX_OPCODE_COLLECT &&
                     collect->nr_srcs == 2 &&
                     scalar_publication_source(collect->src[0]) &&
                     scalar_publication_source(collect->src[1]) &&
                     (!I->shadow || scalar_publication_source(I->src[5]));
         if (eligible) {
            agx_index x = collect->src[0], y = collect->src[1];
            agx_index reference = I->shadow ? I->src[5] : agx_null();
            unsigned match = count;
            for (unsigned j = 0; j < count; ++j) {
               if (same_register_value(available[j].x, x) &&
                   same_register_value(available[j].y, y) &&
                   same_register_value(available[j].reference, reference)) {
                  match = j;
                  break;
               }
            }
            uint32_t value;
            if (match < count) {
               const struct scalar_publication hit = available[match];
               value = hit.value;
               memmove(&available[match], &available[match + 1],
                       (count - match - 1) * sizeof(*available));
               available[count - 1] = hit;
            } else {
               const unsigned cost = I->shadow ? 8 : 4;
               while (reserved + cost > AGX3_STAGING_HALFS) {
                  reserved -= available[0].cost;
                  memmove(available, available + 1, --count * sizeof(*available));
               }
               agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
               agx_publish_texture_scalar_agx3(&b, x, y, reference, ++next);
               value = next;
               available[count++] = (struct scalar_publication){
                  x, y, reference, value, cost,
               };
               reserved += cost;
            }
            I->src[0] = agx_get_vec_index(value, AGX_SIZE_32, I->shadow ? 3 : 2);
            I->src[0].type = AGX_INDEX_RESOURCE;
            if (I->shadow)
               I->src[5] = agx_null();
            progress = true;
         }
         if (!ssa_preserves_publications(I))
            count = reserved = 0;
      }
   }
   ralloc_free(defs);
   return progress;
}

static bool
fma_staging_producer(const agx_instr *I, bool immutable_uniforms)
{
   if (!I || I->op != AGX_OPCODE_FFMA || I->nr_srcs != 3 || I->nr_dests != 1 ||
       I->dest[0].type != AGX_INDEX_NORMAL || I->dest[0].size != AGX_SIZE_32 ||
       agx_channels(I->dest[0]) != 1 || I->saturate || I->agx3_tex_coord_stage)
      return false;
   unsigned special = 0;
   for (unsigned i = 0; i < 3; ++i) {
      const agx_index src = I->src[i];
      const bool immediate = src.type == AGX_INDEX_IMMEDIATE;
      if (src.size != (immediate ? AGX_SIZE_16 : AGX_SIZE_32) ||
          agx_channels(src) != 1 || src.memory || src.abs || src.neg)
         return false;
      if (immediate) {
         if (i == 2 || src.value >= 256)
            return false;
         special++;
      } else if (src.type == AGX_INDEX_UNIFORM) {
         if (!immutable_uniforms || i == 2 || src.value >= 256 ||
             (src.value & 1))
            return false;
         special++;
      } else if (src.type != AGX_INDEX_NORMAL)
         return false;
   }
   return special <= 1;
}

bool
agx3_fuse_fma_texture_publications(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       ctx->key->fs.ignore_tib_dependencies || list_length(&ctx->blocks) != 1)
      return false;
   agx_instr **defs = rzalloc_array(ctx, agx_instr *, ctx->alloc);
   unsigned *uses = rzalloc_array(ctx, unsigned, ctx->alloc);
   if (!defs || !uses)
      return false;
   bool immutable = true;
   agx_foreach_instr_global(ctx, I) {
      if (I->agx3_tex_coord_stage)
         return false;
      immutable &= I->op != AGX_OPCODE_UNIFORM_STORE &&
                   I->op != AGX_OPCODE_UNIFORM_FADD_AGX3 &&
                   I->op != AGX_OPCODE_UNIFORM_FMUL_AGX3 &&
                   I->op != AGX_OPCODE_UNIFORM_FFMA_AGX3;
      agx_foreach_ssa_src(I, i)
         uses[I->src[i].value]++;
      agx_foreach_ssa_dest(I, i)
         defs[I->dest[i].value] = I;
   }
   bool measured = false;
   unsigned demand = 0;
   agx_foreach_instr_global(ctx, publication) {
      if (publication->op != AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3 ||
          publication->nr_srcs != 3 || !agx_is_null(publication->src[2]))
         continue;
      agx_instr *producer[2];
      bool valid = true;
      for (unsigned c = 0; c < 2; ++c) {
         const agx_index src = publication->src[c];
         if (src.type != AGX_INDEX_NORMAL || src.size != AGX_SIZE_32 ||
             agx_channels(src) != 1 || src.abs || src.neg || src.memory ||
             uses[src.value] != 1 ||
             !fma_staging_producer(defs[src.value], immutable)) {
            valid = false;
            break;
         }
         producer[c] = defs[src.value];
      }
      if (!valid || producer[0] == producer[1])
         continue;
      if (!measured) {
         agx_compute_liveness(ctx);
         demand = agx_calc_register_demand(ctx, false);
         measured = true;
      }
      struct list_head *next[2] = {producer[0]->link.next,
                                   producer[1]->link.next};
      unsigned earlier = 0;
      agx_foreach_instr_global(ctx, I) {
         if (I == producer[0] || I == producer[1]) {
            earlier = I == producer[1];
            break;
         }
      }
      for (unsigned c = 0; c < 2; ++c) {
         list_del(&producer[c]->link);
         list_addtail(&producer[c]->link, &publication->link);
      }
      agx_compute_liveness(ctx);
      if (agx_calc_register_demand(ctx, false) > demand) {
         list_del(&producer[0]->link);
         list_del(&producer[1]->link);
         list_addtail(&producer[earlier ^ 1]->link, next[earlier ^ 1]);
         list_addtail(&producer[earlier]->link, next[earlier]);
         agx_compute_liveness(ctx);
         continue;
      }
      for (unsigned c = 0; c < 2; ++c) {
         producer[c]->agx3_tex_coord_stage = true;
         producer[c]->agx3_tex_coord_component = c;
      }
      producer[0]->agx3_tex_coord_start = true;
      publication->agx3_tex_coords_staged = true;
      return true;
   }
   return false;
}
