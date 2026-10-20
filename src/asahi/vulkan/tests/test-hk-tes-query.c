/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_descriptor_set_layout.h"
#include "hk_shader.h"
#include "hk_native_vertex_template.h"
#include "asahi/lib/agx_vdm_state_update_core.h"
#include "asahi/compiler/agx_nir.h"
#include "asahi/compiler/agx_nir_gen1_ray_query.h"
#include "compiler/nir/nir_builder.h"
#include "poly/nir/poly_nir.h"
#include "vk_pipeline.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
 fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

/* Descriptor templates and ordinary writes have the same shader-visible
 * layout; push descriptors differ only in layout flags at this stage. Keep
 * both set placements and both final execution stages under test. */
static void
check_descriptor(bool push, unsigned set, unsigned binding, bool hardware)
{
   struct hk_descriptor_set_layout *layout = calloc(1, sizeof(*layout) +
      (binding + 1) * sizeof(layout->binding[0]));
   CHECK(layout);
   layout->vk.flags = push ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT : 0;
   layout->descriptor_generation = 9;
   layout->binding_count = binding + 1;
   layout->binding[binding] = (struct hk_descriptor_set_binding_layout){
      .type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
      .array_size = 1, .stride = 8, .offset = 24,
   };
   struct vk_descriptor_set_layout *layouts[2] = {0};
   layouts[set] = &layout->vk;
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_TESS_EVAL,
      &agx_nir_options, "TES acceleration structure descriptor");
   b.shader->info.io_lowered = true;
   b.shader->info.tess._primitive_mode = TESS_PRIMITIVE_TRIANGLES;
   b.shader->info.tess.tcs_vertices_out = 3;
   nir_def *idx = nir_vulkan_resource_index(&b, 1, 64, nir_imm_int(&b, 0),
      .desc_set = set, .binding = binding,
      .desc_type = nir_descriptor_type_acceleration_structure,
      .resource_type = nir_resource_type_acceleration_structure);
   nir_def *as = nir_load_vulkan_descriptor(&b, 1, 64, idx,
      .desc_type = nir_descriptor_type_acceleration_structure);
   nir_variable *query = nir_local_variable_create(b.impl,
      glsl_uint64_t_type(), "query");
   query->data.ray_query = true;
   nir_def *rq = &nir_build_deref_var(&b, query)->def;
   nir_rq_initialize(&b, rq, as, nir_imm_int(&b, 0), nir_imm_int(&b, 255),
      nir_imm_vec3(&b, 0, 0, 0), nir_imm_float(&b, 0),
      nir_imm_vec3(&b, 0, 0, 1), nir_imm_float(&b, 9));
   nir_push_loop(&b);
   nir_break_if(&b, nir_inot(&b, nir_rq_proceed(&b, 1, rq)));
   nir_def *distance = nir_rq_load(&b, 1, 32, rq,
      .ray_query_value = nir_ray_query_value_intersection_t, .committed = false);
   nir_store_global(&b, distance, nir_imm_int64(&b, 0x10004000000ull),
      .align_mul = 4, .write_mask = 1);
   nir_pop_loop(&b, NULL);
   nir_shader_gather_info(b.shader, b.impl);
   CHECK(hk_nir_lower_descriptors(b.shader, &vk_robustness_disabled,
      set + 1, layouts, false, 0, false, false));
   const struct agx_gen1_ray_query_layout rq_layout = {
      .payload_uniform = 118, .bvh_uniform = 96, .status_uniform = 128,
      .context_uniform = 160, .as_table_uniform = 164, .as_table_first = 1,
      .launch_header = 0x210000, .instance_traversal = true,
      .candidates = true, .graphics_stage = hardware, .compute_kernel = true,
   };
   CHECK(agx_nir_lower_gen1_ray_query(b.shader, &rq_layout));
   CHECK(poly_nir_lower_tes(b.shader, hardware));
   NIR_PASS(_, b.shader, nir_opt_constant_folding);
   unsigned descriptor_reads = 0, set_reads = 0, intersections = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         CHECK(!nir_intrinsic_is_ray_query(intr->intrinsic));
         CHECK(intr->intrinsic != nir_intrinsic_load_vulkan_descriptor);
         if (intr->intrinsic == nir_intrinsic_load_descriptor_set_agx) {
            CHECK(nir_intrinsic_desc_set(intr) == set);
            ++set_reads;
         }
         if (intr->intrinsic == nir_intrinsic_load_global_constant_offset) {
            CHECK(intr->def.num_components == 1 && intr->def.bit_size == 64);
            CHECK(nir_src_is_const(intr->src[1]));
            CHECK(nir_src_as_uint(intr->src[1]) == 24);
            ++descriptor_reads;
         }
         intersections += intr->intrinsic == nir_intrinsic_intersect_ray_gen1_agx;
      }
   }
   CHECK(descriptor_reads == 1 && set_reads == 1 && intersections == 1);
   CHECK(b.shader->info.stage == (hardware ? MESA_SHADER_VERTEX : MESA_SHADER_TESS_EVAL));
   nir_validate_shader(b.shader, "TES AS descriptor lowering");
   ralloc_free(b.shader);
   free(layout);
}

static void
check_template(bool discard, bool writes, bool xfb)
{
   struct agx_vdm_state_update_draw_info info = {
      .source_layout = AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT,
      .draw_path = AGX_VDM_STATE_UPDATE_DRAW_DIRECT_INDEXED,
      .rasterization_enabled = hk_native_vertex_template_enabled(discard, writes, xfb),
      .raw_provoking_selector = 0,
      .template_info = {.pipeline = 0x10000000000ull,
         .raw_active_context_selector = UINT32_MAX},
   };
   struct agx_vdm_state_update encoded;
   struct agx_vdm_state_update_info decoded;
   CHECK(agx_vdm_state_update_draw_pack(AGX_CHIP_G17P, &info, &encoded));
   CHECK(agx_vdm_state_update_unpack(AGX_CHIP_G17P, encoded.bytes, encoded.length, &decoded));
   /* The pre-HAL300 discard template sets 0x30 in the second output word.
    * A shader with stores must retain the live template independently of
    * its API stage and of whether those stores came from transform feedback. */
   CHECK((decoded.raw_vertex_outputs[1] & 0x30) ==
         ((discard && !writes && !xfb) ? 0x30 : 0));
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   unsigned cases = 0;
   for (unsigned push = 0; push < 2; ++push)
      for (unsigned set = 0; set < 2; ++set)
         for (unsigned binding = 0; binding < 2; ++binding)
            for (unsigned hw = 0; hw < 2; ++hw) {
               check_descriptor(push, set, binding, hw);
               ++cases;
            }
   for (unsigned discard = 0; discard < 2; ++discard)
      for (unsigned writes = 0; writes < 2; ++writes)
         for (unsigned xfb = 0; xfb < 2; ++xfb)
            check_template(discard, writes, xfb);
   glsl_type_singleton_decref();
   printf("TES AS descriptor lowering controls: %u passed\n", cases);
   return 0;
}
