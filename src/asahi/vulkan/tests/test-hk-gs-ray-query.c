/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_nir_lower_gs.h"
#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_nir_gen1_ray_query.h"
#include "nir_builder.h"
#include "poly/nir/poly_nir.h"
#include <stdio.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return false; \
} } while (0)

static bool
check_variant(nir_shader *nir, bool raster, bool require_query, bool instances)
{
   if (!nir)
      return !require_query;
   nir_validate_shader(nir, "GS native query split");
   unsigned intersects = 0, context_footers = 0, as_loads = 0, table_stores = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         CHECK(!nir_intrinsic_is_ray_query(intr->intrinsic));
         if (intr->intrinsic == nir_intrinsic_intersect_ray_gen1_agx)
            ++intersects;
         if (intr->intrinsic == nir_intrinsic_store_scs_agx &&
             nir_intrinsic_base(intr) == 0x6068) {
            CHECK(intr->src[0].ssa->num_components == (raster ? 4 : 2));
            ++context_footers;
         }
         if (intr->intrinsic == nir_intrinsic_load_global_constant &&
             intr->def.bit_size == 64 && nir_src_is_const(intr->src[0]) &&
             nir_src_as_uint(intr->src[0]) == 0x10003000000ull)
            ++as_loads;
         if (intr->intrinsic == nir_intrinsic_store_global &&
             intr->src[0].ssa->bit_size == 64)
            ++table_stores;
      }
   }
   CHECK(nir->info.ray_queries == 0);
   if (require_query) {
      CHECK(intersects && context_footers && as_loads);
      if (!instances)
         CHECK(table_stores);
   }
   return true;
}

static bool
check_split(bool dynamic_count, bool instances)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "GS AS descriptor query");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = 1;
   b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   b.shader->info.outputs_written = VARYING_BIT_POS;
   nir_variable *query = nir_local_variable_create(
      b.impl, glsl_uint64_t_type(), "query");
   query->data.ray_query = true;
   nir_def *rq = &nir_build_deref_var(&b, query)->def;
   /* The bare 64-bit address produced by AS descriptor lowering, loaded at
    * execution time so it observes descriptor updates between draws. */
   nir_def *as = nir_load_global_constant(&b, 1, 64,
      nir_imm_int64(&b, 0x10003000000ull), .align_mul = 8);
   nir_rq_initialize(&b, rq, as, nir_imm_int(&b, 1), nir_imm_int(&b, 255),
      nir_imm_vec3(&b, 0, 0, -1), nir_imm_float(&b, 0),
      nir_imm_vec3(&b, 0, 0, 1), nir_imm_float(&b, 10));
   nir_push_loop(&b);
   nir_break_if(&b, nir_inot(&b, nir_rq_proceed(&b, 1, rq)));
   nir_pop_loop(&b, NULL);
   nir_def *type = nir_rq_load(&b, 1, 32, rq,
      .ray_query_value = nir_ray_query_value_intersection_type,
      .committed = true);
   if (dynamic_count)
      nir_push_if(&b, nir_ine_imm(&b, type, 0));
   nir_store_output(&b, nir_vec4(&b, nir_u2f32(&b, type),
      nir_imm_float(&b, 0), nir_imm_float(&b, 0), nir_imm_float(&b, 1)),
      nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics.location = VARYING_SLOT_POS);
   nir_emit_vertex(&b, 0);
   if (dynamic_count)
      nir_pop_if(&b, NULL);
   nir_shader_gather_info(b.shader, b.impl);
   CHECK(b.shader->info.ray_queries == 1);
   agx_preprocess_nir_g17p(b.shader);
   const struct agx_gen1_ray_query_layout layout = {
      .payload_uniform = instances ? 118 : 124,
      .bvh_uniform = 96,
      .status_uniform = 128,
      .context_uniform = 160,
      .as_table_uniform = 164,
      .as_table_first = 1,
      .launch_header = instances ? 0x210000 : 0x110000,
      .instance_traversal = instances,
      .candidates = instances,
      .compute_kernel = true,
      .graphics_stage = true,
   };
   nir_shader *count = NULL, *rast = NULL, *pre_gs = NULL;
   struct poly_gs_info info;
   CHECK(hk_nir_lower_gs(b.shader, &count, &rast, &pre_gs, &info, &layout));
   CHECK(rast && rast->info.stage == MESA_SHADER_VERTEX);
   CHECK(!!count == dynamic_count);
   CHECK(check_variant(rast, true, true, instances));
   CHECK(check_variant(b.shader, false, dynamic_count, instances));
   CHECK(check_variant(count, false, dynamic_count, instances));
   ralloc_free(b.shader);
   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre_gs);
   printf("GS query split: dynamic=%u instances=%u PASS\n", dynamic_count, instances);
   return true;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool valid = true;
   for (unsigned dynamic = 0; dynamic < 2; ++dynamic)
      for (unsigned instances = 0; instances < 2; ++instances)
         valid &= check_split(dynamic, instances);
   glsl_type_singleton_decref();
   return valid ? 0 : 1;
}
