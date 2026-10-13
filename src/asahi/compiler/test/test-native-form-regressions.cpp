#include "agx3_alu_core.h"
#include "agx3_pack.h"
#include "agx3_stage_core.h"
#include "agx3_tex_core.h"
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
#include "asahi/layout/layout.h"
}
#include <gtest/gtest.h>
#include <vector>

TEST(NativeForms, BoundedUnaryRegistersUnderPressure)
{
   for (agx_opcode op : {AGX_OPCODE_TRUNC, AGX_OPCODE_DFDX, AGX_OPCODE_DFDY}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index live[64];
      for (unsigned i = 0; i < 64; ++i)
         live[i] = agx_preload(b, agx_register(2 * i, AGX_SIZE_32));
      agx_index input = agx_mov_imm(b, 32, 0x40500000);
      agx_index output = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *unary = agx_alloc_instr(b, op, 1, 1);
      unary->src[0] = input;
      unary->dest[0] = output;
      agx_builder_insert(&b->cursor, unary);
      agx_index sum = output;
      for (unsigned i = 0; i < 64; ++i)
         sum = agx_fadd(b, sum, live[i]);
      agx_export(b, sum, 8);
      agx_stop(b);
      agx_ra(b->shader);
      EXPECT_LT(unary->src[0].value, 128u) << op;
      EXPECT_LT(unary->dest[0].value, 128u) << op;
      unsigned high = 0;
      agx_foreach_instr_global(b->shader, I) {
         agx_foreach_dest(I, d)
            high += I->dest[d].type == AGX_INDEX_REGISTER && I->dest[d].value >= 128;
      }
      EXPECT_GT(high, 0u) << "Unrelated live values must retain the upper bank";
      ralloc_free(mem);
   }
}

TEST(NativeForms, NegatedSqrtStepReachesPacker)
{
   for (bool absolute : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index src = agx_register(112, AGX_SIZE_32);
      src.neg = true;
      src.abs = absolute;
      src.cache = true;
      agx_instr *step = agx_alloc_instr(b, AGX_OPCODE_SRSQRT, 1, 1);
      step->src[0] = src;
      step->dest[0] = agx_register(122, AGX_SIZE_32);
      agx_builder_insert(&b->cursor, step);
      agx_stop(b);
      util_dynarray bytes;
      util_dynarray_init(&bytes, mem);
      EXPECT_TRUE(agx3_pack_binary(b->shader, &bytes));
      EXPECT_GT(bytes.size, 4u);
      ralloc_free(mem);
   }
}

TEST(NativeForms, HalfModifiedMultiplyAndSelect)
{
   for (bool select : {false, true}) {
      SCOPED_TRACE(select);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "half modified operands");
      nir_def *input = agx_test_load_preamble(&b, 4, 32, 4);
      nir_def *x = nir_f2f16(&b, nir_channel(&b, input, 0));
      nir_def *result;
      if (select) {
         nir_def *cmp = agx_test_load_preamble(&b, 2, 16, 12);
         x = nir_channel(&b, cmp, 0);
         nir_def *y = nir_channel(&b, cmp, 1);
         result = nir_bcsel(&b, nir_flt(&b, x, nir_fneg(&b, y)),
                           nir_channel(&b, input, 2),
                           nir_channel(&b, input, 3));
      } else {
         result = nir_f2f32(
            &b, nir_fmul(&b, nir_fabs(&b, x), nir_imm_floatN_t(&b, 0.5, 16)));
      }
      agx_test_store_local_pixel_agx(
         &b, nir_vec4(&b, nir_channel(&b, input, 0), result, result, result),
         nir_imm_intN_t(&b, 0xff, 16), nir_undef(&b, 2, 16), 0, 0xf,
         PIPE_FORMAT_R8_UNORM);
      b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.no_preamble = true;
      key.reserved_preamble = 16;
      key.fs.nr_samples = 1;
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_GT(part.info.main_size, 0u);
      free(part.binary);
   }
}

TEST(NativeForms, VertexImageLoadRegisterLod)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "vertex image load register LOD");
   nir_def *id = nir_load_vertex_id(&b);
   nir_def *coord = nir_vec4(&b, id, id, id, nir_imm_int(&b, 0));
   nir_def *value = nir_image_load(
      &b, 4, 32, nir_imm_intN_t(&b, 3, 16), coord, nir_imm_int(&b, 0),
      agx_test_load_preamble(&b, 1, 16, 0));
   nir_intrinsic_instr *load = nir_def_as_intrinsic(value);
   nir_intrinsic_set_image_dim(load, GLSL_SAMPLER_DIM_3D);
   nir_intrinsic_set_format(load, PIPE_FORMAT_R32G32B32A32_UINT);
   nir_intrinsic_set_dest_type(load, nir_type_uint32);
   for (unsigned c = 0; c < 4; ++c) {
      nir_store_uvs_agx(&b, nir_imm_float(&b, c == 3 ? 1.0f : 0.0f),
                       nir_imm_intN_t(&b, c, 16));
      nir_store_uvs_agx(&b, nir_channel(&b, value, c),
                       nir_imm_intN_t(&b, 4 + c, 16));
   }
   b.shader->info.outputs_written = VARYING_BIT_POS |
                                   BITFIELD64_BIT(VARYING_SLOT_VAR0);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.no_preamble = true;
   key.reserved_preamble = 64;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   EXPECT_GT(part.info.main_size, 0u);
   free(part.binary);
}

TEST(NativeForms, TexturePublicationWaitPreservesUnrelatedResult)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->max_reg = 31;
   agx_index coords = agx_register(10, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_device_load_to(b, agx_register(0, AGX_SIZE_32),
                     agx_register(28, AGX_SIZE_64), agx_immediate(0),
                     AGX_FORMAT_I32, 1, 0, false);
   agx_device_load_to(b, coords, agx_register(28, AGX_SIZE_64),
                     agx_immediate(4), AGX_FORMAT_I32, 3, 0, false);
   agx_publish_texture_agx3(b, coords, agx_null(), 1);
   agx_index resource = coords;
   resource.type = AGX_INDEX_RESOURCE;
   resource.value = 1;
   agx_index result = agx_register(20, AGX_SIZE_32);
   result.channels_m1 = 3;
   agx_texture_sample_to(b, result, resource, agx_immediate(0),
                        agx_immediate(0), agx_register(24, AGX_SIZE_64),
                        agx_register(26, AGX_SIZE_64), agx_null(), AGX_DIM_2D,
                        AGX_LOD_MODE_LOD_MIN, 15, false, false, false,
                        AGX_GATHER_NONE, false);
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   uint8_t carrier[AGX3_ALU_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_alu_band_identity_wait(AGX_ISA_ENCODING_AGX3B, 10,
                                         2, carrier, &len));
   const uint8_t *bytes = (const uint8_t *)binary.data;
   bool found = false;
   for (unsigned i = 0; i + len <= binary.size; ++i)
      found |= memcmp(bytes + i, carrier, len) == 0;
   EXPECT_TRUE(found);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(NativeForms, MinmaxReusesUniformCopiedForComparison)
{
   for (agx_fcond cond : {AGX_FCOND_LTN, AGX_FCOND_GTN}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      agx_instr *I = agx_alloc_instr(b, AGX_OPCODE_FCMPSEL, 1, 4);
      I->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
      I->src[0] = I->src[2] = agx_uniform(8, AGX_SIZE_32);
      I->src[1] = I->src[3] = agx_immediate(176); /* -1.0 minifloat */
      I->fcond = cond;
      agx_builder_insert(&b->cursor, I);
      agx_lower_uniform_sources(b->shader);
      EXPECT_EQ(I->src[0].type, AGX_INDEX_NORMAL);
      EXPECT_TRUE(agx_is_equiv(I->src[0], I->src[2]));
      ralloc_free(mem);
   }
}

TEST(NativeForms, AtomicVertexOutputsReachTerminalBundle)
{
   for (bool position : {false, true}) {
      SCOPED_TRACE(position);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "atomic vertex outputs");
      nir_def *address = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *value = nir_global_atomic_agx(
         &b, 32, address, nir_imm_int(&b, 0), nir_imm_int(&b, 1));
      nir_intrinsic_set_atomic_op(nir_def_as_intrinsic(value), nir_atomic_op_iadd);
      nir_store_uvs_agx(&b, value, nir_imm_intN_t(&b, 4, 16));
      nir_def *second = nir_global_atomic_agx(
         &b, 32, address, nir_imm_int(&b, 4), nir_imm_int(&b, -1));
      nir_intrinsic_set_atomic_op(nir_def_as_intrinsic(second), nir_atomic_op_iadd);
      nir_store_uvs_agx(&b, nir_iadd(&b, value, second),
                       nir_imm_intN_t(&b, 5, 16));
      if (position) {
         for (unsigned c = 0; c < 4; ++c)
            nir_store_uvs_agx(&b, nir_imm_float(&b, c == 3 ? 1.0 : 0.0),
                             nir_imm_intN_t(&b, c, 16));
      }
      b.shader->info.outputs_written = BITFIELD64_BIT(VARYING_SLOT_VAR0) |
                                     (position ? VARYING_BIT_POS : 0);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.no_preamble = true;
      key.reserved_preamble = 4;
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_GT(part.info.main_size, 0u);
      free(part.binary);
      ralloc_free(b.shader);
   }
}

TEST(NativeForms, RectangleImplicitLodOutsideFragment)
{
   for (mesa_shader_stage stage : {MESA_SHADER_VERTEX, MESA_SHADER_COMPUTE,
                                   MESA_SHADER_FRAGMENT}) {
      for (bool sampler_bias : {false, true}) {
         SCOPED_TRACE(stage);
         SCOPED_TRACE(sampler_bias);
         nir_builder b = nir_builder_init_simple_shader(
            stage, &agx_nir_options, "rectangle implicit lod");
         nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
         tex->op = nir_texop_tex;
         tex->sampler_dim = GLSL_SAMPLER_DIM_RECT;
         tex->dest_type = nir_type_float32;
         tex->coord_components = 2;
         tex->src[0] = nir_tex_src_for_ssa(
            nir_tex_src_coord, nir_imm_vec2(&b, 3.0, 5.0));
         nir_def_init(&tex->instr, &tex->def, 4, 32);
         nir_builder_instr_insert(&b, &tex->instr);
         agx_nir_lower_g17p_texture_early(b.shader, sampler_bias);
         EXPECT_EQ(tex->sampler_dim, GLSL_SAMPLER_DIM_RECT);
         EXPECT_GE(nir_tex_instr_src_index(tex, nir_tex_src_coord), 0);
         if (stage == MESA_SHADER_FRAGMENT) {
            EXPECT_EQ(tex->op, sampler_bias ? nir_texop_txb : nir_texop_tex);
         } else {
            EXPECT_EQ(tex->op, nir_texop_txl);
            EXPECT_LT(nir_tex_instr_src_index(tex, nir_tex_src_bias), 0);
            int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
            ASSERT_GE(lod, 0);
            EXPECT_EQ(nir_src_as_float(tex->src[lod].src), 0.0);
         }
         nir_validate_shader(b.shader, "rectangle level zero");
         ralloc_free(b.shader);
      }
   }
}

TEST(NativeForms, ImageFenceWaitsRotatingWriteSlots)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index data = agx_register(16, AGX_SIZE_32);
   data.channels_m1 = 3;
   agx_index coord = agx_register(24, AGX_SIZE_32);
   coord.channels_m1 = 1;
   for (unsigned i = 0; i < 3; ++i) {
      agx_image_write(b, data, coord, agx_zero(), agx_uniform(8, AGX_SIZE_64),
                      agx_register(28, AGX_SIZE_32), AGX_DIM_2D, true);
      agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT);
   }
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));

   agx3_encoded_scoped_barrier fence;
   ASSERT_TRUE(agx3_pack_scoped_barrier(AGX_ISA_ENCODING_AGX3B,
                                       AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT,
                                       &fence));
   const uint8_t *bytes = (const uint8_t *)binary.data;
   unsigned fences = 0;
   for (unsigned i = 0; i + fence.length <= binary.size; ++i) {
      if (memcmp(bytes + i, fence.bytes, fence.length))
         continue;
      /* The first write uses the fence's embedded slot1 wait. Later writes
       * rotate to slots2/3 and must finish before the cache fence, not after. */
      if (fences > 0) {
         uint8_t carrier[AGX3_ALU_MAX_LENGTH], len = 0;
         ASSERT_TRUE(agx3_alu_band_identity_wait(
            AGX_ISA_ENCODING_AGX3B, 0, 1u << fences, carrier, &len));
         ASSERT_GE(i, len);
         EXPECT_EQ(memcmp(bytes + i - len, carrier, len), 0) << fences;
      }
      ++fences;
      i += fence.length - 1;
   }
   EXPECT_EQ(fences, 3u);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(NativeForms, ImmediateCubeGathersReserveCemScratch)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->max_reg = 31;

   /* This is the hardware reproducer's allocated shape. The red gather
    * remains live in r0 while the remaining gathers compute their cube
    * coordinates. The old immediate-descriptor path used r0 as CEM scratch
    * and overwrote red with the next face calculation. */
   agx_index coords = agx_register(16, AGX_SIZE_32);
   coords.channels_m1 = 3;
   const unsigned destinations[] = {0, 8, 24, 16};
   for (unsigned component = 0; component < 4; ++component) {
      agx_index dst = agx_register(destinations[component], AGX_SIZE_32);
      dst.channels_m1 = 3;
      agx_texture_sample_to(b, dst, coords, agx_immediate(0),
                           agx_immediate(0), agx_immediate(0),
                           agx_immediate(1), agx_null(), AGX_DIM_CUBE_ARRAY,
                           AGX_LOD_MODE_AUTO_LOD, 15, false, false, false,
                           (enum agx_gather)(AGX_GATHER_R + 2 * component), false);
   }
   agx_stop(b);

   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_GE(b->shader->max_reg, 39u);

   /* Check the actual emitted CEM destination, not just the reservation.
    * All four axis-zero operations must write above the live result bank. */
   uint8_t cem[AGX3_TEX_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_tex_cube_cem_calc_to(AGX_ISA_ENCODING_AGX3B, 16, 18, 20,
                                         32, 0, false, cem, &len));
   const uint8_t *bytes = (const uint8_t *)binary.data;
   unsigned count = 0;
   for (unsigned i = 0; i + len <= binary.size; ++i)
      count += memcmp(bytes + i, cem, len) == 0;
   EXPECT_EQ(count, 4u);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}


TEST(NativeForms, LateCoverageStaysAfterImageSideEffects)
{
   for (bool depth : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index data = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_device_load_to(b, data, agx_uniform(8, AGX_SIZE_64), agx_immediate(0),
                         AGX_FORMAT_I32, 15, 0, false);
      agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_device_load_to(b, coord, agx_uniform(8, AGX_SIZE_64), agx_immediate(4),
                         AGX_FORMAT_I32, 3, 0, false);
      agx_instr *store = agx_image_write(
         b, data, coord, agx_zero(), agx_uniform(12, AGX_SIZE_64),
         agx_mov_imm(b, 32, 1), AGX_DIM_2D, true);
      agx_index mask = agx_mov_imm(b, 16, 1);
      agx_index coverage[8];
      for (unsigned i = 0; i < 8; ++i)
         coverage[i] = agx_get_coverage_1x_agx3(b, 16);
      agx_index word = coverage[0];
      for (unsigned i = 1; i < 8; ++i)
         word = agx_iadd(b, word, coverage[i], 0);
      word = agx_iadd(b, mask, word, 4);
      agx_index wide = agx_temp(b->shader, AGX_SIZE_32);
      agx_mov_to(b, wide, word);
      agx_instr *tests = depth ? agx_zs_emit_agx3(b, wide, 1)
                              : agx_sample_mask_agx3(b, mask, wide, 2);
      agx_export(b, data, 8);
      agx_pressure_schedule(b->shader);
      bool saw_store = false;
      agx_foreach_instr_global(b->shader, I) {
         saw_store |= I == store;
         if (I == tests) {
            EXPECT_TRUE(saw_store) << "Native late tests preceded imageStore";
         }
      }
      ralloc_free(mem);
   }
}

TEST(NativeForms, VertexOutputsAllowTrailingMemoryFence)
{
   for (unsigned count : {5u, 6u, 11u}) {
      SCOPED_TRACE(count);
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_VERTEX;
      for (unsigned i = 0; i < count; ++i)
         agx_st_vary(b, agx_immediate(i), agx_register(2 * i, AGX_SIZE_32));
      agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE_FENCE);
      agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT);
      agx_stop(b);
      util_dynarray bytes;
      util_dynarray_init(&bytes, mem);
      EXPECT_TRUE(agx3_pack_binary(b->shader, &bytes));
      EXPECT_GT(bytes.size, 0u);
      ralloc_free(mem);
   }
}

TEST(NativeForms, CubeArrayShadowZeroLodUsesNativeDescriptors)
{
   for (mesa_shader_stage stage : {MESA_SHADER_VERTEX, MESA_SHADER_TESS_CTRL,
                                MESA_SHADER_TESS_EVAL, MESA_SHADER_GEOMETRY,
                                MESA_SHADER_FRAGMENT, MESA_SHADER_COMPUTE}) {
      for (float lod : {0.0f, 1.0f}) {
         SCOPED_TRACE(::testing::Message() << "stage=" << stage << " lod=" << lod);
         nir_builder b = nir_builder_init_simple_shader(
            stage, &agx_nir_options, "cube array shadow native descriptors");
         nir_def *texture = nir_bindless_image_agx(&b, nir_imm_int(&b, 0));
         nir_def *sampler = nir_bindless_sampler_agx(
            &b, nir_imm_int(&b, 32), nir_imm_intN_t(&b, 5, 16));
         nir_tex_instr *tex = nir_tex_instr_create(b.shader, 5);
         tex->op = nir_texop_txl;
         tex->sampler_dim = GLSL_SAMPLER_DIM_CUBE;
         tex->is_array = true;
         tex->is_shadow = true;
         tex->is_new_style_shadow = true;
         tex->coord_components = 4;
         tex->dest_type = nir_type_float32;
         tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1,
                                         nir_imm_vec4(&b, 1, 0, 0, 0));
         tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                         nir_imm_float(&b, 0.5));
         tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, lod));
         tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, texture);
         tex->src[4] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
         nir_def_init(&tex->instr, &tex->def, 1, 32);
         nir_builder_instr_insert(&b, &tex->instr);
         nir_store_ssbo(&b, &tex->def, nir_imm_int(&b, 0), nir_imm_int(&b, 0));
         EXPECT_EQ(agx_nir_lower_g17p_indexed_samplers(b.shader), lod == 0.0f);
         ralloc_free(b.shader);
      }
   }
}

TEST(NativeForms, EndOfTilePreservesSamplesAndResolvesSameTile)
{
   /* Both outputs read offset zero. Their descriptors are distinct, and
    * only the second descriptor names a single-sample image. Keep all four
    * dimension combinations as controls; no host check establishes pixels. */
   for (unsigned resolves = 0; resolves < 4; ++resolves) {
      SCOPED_TRACE(resolves);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "EOT store and resolve");
      nir_def *xy = nir_trim_vector(&b, nir_load_global_invocation_id(&b, 32), 2);
      for (unsigned pbe = 0; pbe < 2; ++pbe) {
         nir_intrinsic_instr *intr = nir_image_store_block_agx(
            &b, nir_imm_intN_t(&b, pbe, 16), nir_imm_int(&b, 0), xy);
         nir_intrinsic_set_format(intr, (enum pipe_format)AIL_ISA_FORMAT_U8NORM);
         nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
         nir_intrinsic_set_image_array(intr, false);
         nir_intrinsic_set_explicit_coord(intr, true);
      }
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.no_preamble = true;
      key.g17p_rt0_eot_multisample = true;
      key.g17p_mrt_eot_count = 2;
      key.g17p_mrt_eot_resolve_mask = resolves;
      key.g17p_mrt_eot_selectors[0] = 0x75;
      key.g17p_mrt_eot_selectors[1] = 0x75;
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      bool ok = agx_compile_shader_nir(b.shader, &key, &part);
      EXPECT_TRUE(ok);
      if (ok) {
         EXPECT_EQ(part.info.binary_size, 120u);
         EXPECT_EQ(part.info.main_size, 120u);
         EXPECT_TRUE(part.info.agx3_staging_known);
         EXPECT_EQ(part.info.agx3_staging_halfs, 16u);
         if (part.info.binary_size == 120) {
            const uint8_t *code = static_cast<const uint8_t *>(part.binary);
            /* Native single-target MS and SS dimension fields, independently
             * selected for each store of the existing dual-target program. */
            for (unsigned pbe = 0; pbe < 2; ++pbe) {
               bool ms = !(resolves & (1u << pbe));
               EXPECT_EQ(code[pbe ? 63 : 43],
                         (pbe ? 0x0c : 0x04) | (ms ? 0x02 : 0x00));
               EXPECT_EQ(code[92 + 12 * pbe + 8], ms ? 0x28 : 0xa8);
               EXPECT_EQ(code[92 + 12 * pbe + 9], ms ? 0x72 : 0x75);
            }
         }
      }
      free(part.binary);
      ralloc_free(b.shader);
   }
}

static agx_shader_info
compile_partial_varying(unsigned components, unsigned bits,
                        glsl_interp_mode mode)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "partial varying cache");
   nir_def *bary = nir_load_barycentric_pixel(&b, 32);
   nir_intrinsic_set_interp_mode(nir_def_as_intrinsic(bary), mode);
   nir_def *input = nir_load_interpolated_input(&b, components, bits, bary,
                                                nir_imm_int(&b, 0));
   nir_intrinsic_instr *load = nir_def_as_intrinsic(input);
   nir_intrinsic_set_component(load, components == 1 ? 3 : 0);
   nir_intrinsic_set_dest_type(
      load, bits == 16 ? nir_type_float16 : nir_type_float32);
   nir_intrinsic_set_io_semantics(
      load, nir_io_semantics{.location = VARYING_SLOT_VAR0});
   nir_def *value = nir_channel(&b, input, components - 1);
   if (bits == 16)
      value = nir_f2f32(&b, value);
   agx_test_store_local_pixel_agx(&b, nir_vec4(&b, value, value, value, value),
                                  nir_imm_intN_t(&b, 0xff, 16),
                                  nir_undef(&b, 2, 16), 0, 0xf,
                                  PIPE_FORMAT_R32_UINT);
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.no_preamble = true;
   key.fs.nr_samples = 1;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   agx_shader_info info = part.info;
   free(part.binary);
   ralloc_free(b.shader);
   return info;
}

TEST(NativeForms, PartialVaryingOmitsUnusedChannels)
{
   for (unsigned bits : {16u, 32u}) {
      for (glsl_interp_mode mode :
           {INTERP_MODE_SMOOTH, INTERP_MODE_NOPERSPECTIVE}) {
         SCOPED_TRACE(bits);
         SCOPED_TRACE(mode);
         agx_shader_info scalar = compile_partial_varying(1, bits, mode);
         agx_shader_info vector = compile_partial_varying(4, bits, mode);
         ASSERT_GT(scalar.main_size, 0u);
         EXPECT_LE(vector.main_size, scalar.main_size + 8);
         EXPECT_LE(vector.nr_gprs, 2 * scalar.nr_gprs);
      }
   }
}

TEST(NativeForms, TextureFmaMatchesMetalStaging)
{
   struct vector {
      uint16_t dest;
      uint8_t slot;
      uint16_t src[3];
      bool live[3], discard[3];
      uint16_t wait;
      uint8_t bytes[16];
      uint8_t length;
   };
   const vector vectors[] = {
      {0,
       1,
       {46, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x09, 0x87, 0x26, 0x2f, 0x01, 0x84, 0x42, 0x02},
       8},
      {4,
       2,
       {48, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x29, 0x87, 0x26, 0x31, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
       10},
      {8,
       2,
       {50, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x49, 0x87, 0x26, 0x33, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
       10},
      {12,
       2,
       {52, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x69, 0x87, 0x26, 0x35, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
       10},
      {2,
       1,
       {46, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x19, 0x83, 0x26, 0x2f, 0x01, 0x80, 0x42, 0x02},
       8},
      {6,
       2,
       {48, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x39, 0x83, 0x26, 0x31, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
       10},
      {10,
       2,
       {50, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x59, 0x83, 0x26, 0x33, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
       10},
      {14,
       2,
       {52, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x79, 0x83, 0x26, 0x35, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
       10},
      {16,
       2,
       {54, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x89, 0x87, 0x26, 0x37, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
       10},
      {20,
       2,
       {56, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0xa9, 0x87, 0x26, 0x39, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
       10},
      {28,
       4,
       {58, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0xe9, 0x87, 0x26, 0x3b, 0x02, 0x84, 0x42, 0x02, 0x00, 0x0c},
       10},
      {36,
       5,
       {62, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x29, 0x87, 0x66, 0x3f, 0x02, 0x84, 0x42, 0x02, 0x00, 0x10},
       10},
      {40,
       5,
       {64, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x49, 0x87, 0x66, 0x41, 0x02, 0x84, 0x42, 0x02, 0x00, 0x10},
       10},
      {42,
       5,
       {64, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x59, 0x83, 0x66, 0x41, 0x02, 0x80, 0x42, 0x02, 0x00, 0x10},
       10},
      {38,
       5,
       {62, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x39, 0x83, 0x66, 0x3f, 0x02, 0x80, 0x42, 0x02, 0x00, 0x10},
       10},
      {44,
       6,
       {60, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x69, 0x87, 0x66, 0x3d, 0x02, 0x84, 0x42, 0x02, 0x00, 0x14},
       10},
      {46,
       6,
       {60, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x79, 0x83, 0x66, 0x3d, 0x02, 0x80, 0x42, 0x02, 0x00, 0x14},
       10},
      {30,
       4,
       {58, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0xf9, 0x83, 0x26, 0x3b, 0x02, 0x80, 0x42, 0x02, 0x00, 0x0c},
       10},
      {22,
       2,
       {56, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0xb9, 0x83, 0x26, 0x39, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
       10},
      {18,
       2,
       {54, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x99, 0x83, 0x26, 0x37, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
       10},
      {32,
       6,
       {66, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x09, 0x87, 0x66, 0x43, 0x02, 0x84, 0x42, 0x02, 0x00, 0x14},
       10},
      {56,
       4,
       {68, 6, 4},
       {0, 0, 0},
       {0, 1, 1},
       0,
       {0xc9, 0x07, 0x6e, 0x45, 0x82, 0x04, 0x42, 0x02, 0x00, 0x0c},
       10},
      {58,
       4,
       {68, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0xd9, 0x83, 0x66, 0x45, 0x02, 0x80, 0x42, 0x02, 0x00, 0x0c},
       10},
      {34,
       6,
       {66, 2, 0},
       {0, 0, 0},
       {0, 1, 1},
       0,
       {0x19, 0x03, 0x6e, 0x43, 0x82, 0x00, 0x42, 0x02, 0x00, 0x14},
       10},
   };
   for (const auto &v : vectors) {
      uint8_t bytes[16], length = 0;
      ASSERT_TRUE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, true, v.dest,
                                        v.slot, v.src, true, v.live, v.discard,
                                        v.wait, bytes, &length));
      ASSERT_EQ(length, v.length);
      EXPECT_EQ(memcmp(bytes, v.bytes, length), 0);
   }
}

TEST(NativeForms, TextureFmaStagingAdmission)
{
   for (unsigned invalid = 0; invalid < 15; ++invalid) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index x = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index y = agx_preload(b, agx_register(2, AGX_SIZE_32));
      agx_index z = agx_preload(b, agx_register(4, AGX_SIZE_32));
      agx_index multiplier = agx_uniform(20, AGX_SIZE_32);
      if (invalid == 12)
         multiplier = agx_preload(b, agx_register(6, AGX_SIZE_32));
      if (invalid == 13)
         multiplier = agx_immediate_f(0.5f);
      if (invalid == 14)
         multiplier = agx_immediate(256);
      if (invalid == 1)
         multiplier.size = AGX_SIZE_16;
      if (invalid == 2)
         y = agx_uniform(22, AGX_SIZE_32);
      if (invalid == 3)
         z = agx_uniform(22, AGX_SIZE_32);
      if (invalid == 4)
         x.neg = true;
      if (invalid == 10)
         multiplier.value = 256;
      if (invalid == 11)
         multiplier.value = 21;
      agx_index u = agx_temp(b->shader, AGX_SIZE_32);
      agx_ffma_to(b, u, x, multiplier, z);
      agx_instr *first =
         agx_last_instr(list_first_entry(&b->shader->blocks, agx_block, link));
      if (invalid == 5)
         first->saturate = true;
      agx_index v = agx_temp(b->shader, AGX_SIZE_32);
      agx_ffma_to(b, v, multiplier, y, z);
      agx_instr *second =
         agx_last_instr(list_first_entry(&b->shader->blocks, agx_block, link));
      if (invalid == 6)
         agx_unit_test(b, u);
      if (invalid == 7) {
         agx_instr *store = agx_alloc_instr(b, AGX_OPCODE_UNIFORM_STORE, 0, 2);
         store->src[0] = z;
         store->src[1] = agx_immediate(20);
         store->mask = 1;
         agx_builder_insert(&b->cursor, store);
      }
      agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_instr *collect = agx_collect_to(b, coord, 2);
      collect->src[0] = u;
      collect->src[1] = v;
      agx_index out = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *sample = agx_texture_sample_to(
         b, out, coord, agx_zero(), agx_zero(), agx_uniform(24, AGX_SIZE_64),
         agx_uniform(28, AGX_SIZE_64), agx_null(), AGX_DIM_2D,
         AGX_LOD_MODE_AUTO_LOD, 0xf, false, false, false, AGX_GATHER_NONE,
         false);
      if (invalid == 8)
         sample->dim = AGX_DIM_3D;
      if (invalid == 9)
         sample->lod_mode = AGX_LOD_MODE_LOD_MIN;
      agx_unit_test(b, out);
      agx3_lower_scalar_texture_publications(b->shader);
      agx_dce(b->shader, true);
      agx_instr *publication = nullptr;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3)
            publication = I;
      }
      const bool progress = agx3_fuse_fma_texture_publications(b->shader);
      EXPECT_EQ(progress, (invalid == 0 || invalid == 9 || invalid == 12 ||
                           invalid == 13))
         << invalid;
      EXPECT_EQ(first->agx3_tex_coord_stage, (invalid == 0 || invalid == 9 ||
                                              invalid == 12 || invalid == 13))
         << invalid;
      EXPECT_EQ(second->agx3_tex_coord_stage, (invalid == 0 || invalid == 9 ||
                                               invalid == 12 || invalid == 13))
         << invalid;
      const bool supported =
         invalid == 0 || invalid == 9 || invalid == 12 || invalid == 13;
      EXPECT_FALSE(sample->agx3_tex_coords_staged);
      if (publication)
         EXPECT_EQ(publication->agx3_tex_coords_staged, supported);
      if (supported) {
         ASSERT_NE(publication, nullptr);
         EXPECT_TRUE(first->agx3_tex_coord_start);
         EXPECT_EQ(first->link.next, &second->link);
         EXPECT_EQ(second->link.next, &publication->link);
         EXPECT_EQ(sample->src[0].type, AGX_INDEX_RESOURCE);
      }
      ralloc_free(mem);
   }
}

TEST(NativeForms, TextureFmaImmediateMatchesMetal)
{
   const uint16_t sources[2][3] = {{32, 6, 4}, {32, 2, 0}};
   const uint8_t expected[2][10] = {
      {0xc9, 0x87, 0x26, 0xa1, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
      {0xd9, 0x83, 0x26, 0xa1, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
   };
   const bool live[3] = {false, true, true};
   const bool discard[3] = {};
   for (unsigned i = 0; i < 2; ++i) {
      uint8_t bytes[16], length = 0;
      ASSERT_TRUE(agx3_stage_fma(AGX_ISA_ENCODING_AGX3B, 24 + 2 * i, 2,
                                 sources[i], AGX3_STAGE_FMA_IMMEDIATE, live,
                                 discard, 0, bytes, &length));
      ASSERT_EQ(length, sizeof(expected[i]));
      EXPECT_EQ(memcmp(bytes, expected[i], length), 0);
   }
   uint8_t bytes[16], length = 0;
   const uint16_t invalid_sources[3] = {256, 2, 0};
   EXPECT_FALSE(agx3_stage_fma(AGX_ISA_ENCODING_AGX3B, 0, 1, invalid_sources,
                               AGX3_STAGE_FMA_IMMEDIATE, live, discard, 0,
                               bytes, &length));
   const bool invalid_live[3] = {true, true, true};
   EXPECT_FALSE(agx3_stage_fma(AGX_ISA_ENCODING_AGX3B, 0, 1, sources[0],
                               AGX3_STAGE_FMA_IMMEDIATE, invalid_live, discard,
                               0, bytes, &length));
}

TEST(NativeForms, TextureFmaStagingPreservesDemand)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index input = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index operands[6];
   for (unsigned i = 0; i < 6; ++i)
      operands[i] = agx_iadd(b, input, agx_immediate(i + 1), 0);
   agx_index x = agx_ffma(b, operands[0], operands[1], operands[2]);
   agx_index y = agx_ffma(b, operands[3], operands[4], operands[5]);
   agx_index live[10];
   for (unsigned i = 0; i < 10; ++i)
      live[i] = agx_iadd(b, input, agx_immediate(i + 10), 0);
   agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, coord, 2);
   collect->src[0] = x;
   collect->src[1] = y;
   agx_index out = agx_temp(b->shader, AGX_SIZE_32);
   agx_texture_sample_to(
      b, out, coord, agx_zero(), agx_zero(), agx_uniform(24, AGX_SIZE_64),
      agx_uniform(28, AGX_SIZE_64), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_AUTO_LOD, 1, false, false, false, AGX_GATHER_NONE, false);
   for (agx_index value : live)
      agx_unit_test(b, value);
   agx_unit_test(b, out);
   ASSERT_TRUE(agx3_lower_scalar_texture_publications(b->shader));
   agx_dce(b->shader, true);
   std::vector<const agx_instr *> order;
   agx_foreach_instr_global(b->shader, I)
      order.push_back(I);
   agx_compute_liveness(b->shader);
   unsigned before = agx_calc_register_demand(b->shader, false);
   EXPECT_FALSE(agx3_fuse_fma_texture_publications(b->shader));
   agx_compute_liveness(b->shader);
   EXPECT_EQ(agx_calc_register_demand(b->shader, false), before);
   unsigned index = 0;
   agx_foreach_instr_global(b->shader, I) {
      EXPECT_FALSE(I->agx3_tex_coords_staged);
      ASSERT_LT(index, order.size());
      EXPECT_EQ(I, order[index++]);
   }
   EXPECT_EQ(index, order.size());
   ralloc_free(mem);
}

TEST(NativeForms, TextureFmaPublicationPacksWithoutCopy)
{
   for (bool loaded : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->max_reg = 31;
      if (loaded)
         agx_device_load_to(b, agx_register(2, AGX_SIZE_32),
                            agx_register(28, AGX_SIZE_64), agx_zero(),
                            AGX_FORMAT_I32, 1, 0, false);
      const agx_index outputs[2] = {
         agx_register(16, AGX_SIZE_32),
         agx_register(18, AGX_SIZE_32),
      };
      agx_instr *x = agx_ffma_to(b, outputs[0], agx_uniform(20, AGX_SIZE_32),
                                 agx_register(2, AGX_SIZE_32),
                                 agx_register(4, AGX_SIZE_32));
      agx_instr *y = agx_ffma_to(b, outputs[1], agx_register(6, AGX_SIZE_32),
                                 agx_uniform(20, AGX_SIZE_32),
                                 agx_register(8, AGX_SIZE_32));
      x->agx3_tex_coord_stage = x->agx3_tex_coord_start = true;
      y->agx3_tex_coord_stage = true;
      y->agx3_tex_coord_component = 1;
      agx_instr *publication = agx_publish_texture_scalar_agx3(
         b, outputs[0], outputs[1], agx_null(), 1);
      publication->agx3_tex_coords_staged = true;
      agx_index resource = agx_get_vec_index(1, AGX_SIZE_32, 2);
      resource.type = AGX_INDEX_RESOURCE;
      agx_index result = agx_register(20, AGX_SIZE_32);
      result.channels_m1 = 3;
      agx_texture_sample_to(b, result, resource, agx_zero(), agx_zero(),
                            agx_uniform(24, AGX_SIZE_64),
                            agx_uniform(28, AGX_SIZE_64), agx_null(),
                            AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false, false,
                            false, AGX_GATHER_NONE, false);
      agx_stop(b);
      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
      const uint8_t *bytes = (const uint8_t *)binary.data;
      const uint16_t src[2][3] = {{20, 2, 4}, {20, 6, 8}};
      const bool hints[3] = {};
      unsigned found[2] = {};
      unsigned publication_slot[2] = {};
      for (unsigned slot = 1; slot <= 6; ++slot) {
         for (unsigned wait = 0; wait < 64; ++wait) {
            for (unsigned c = 0; c < 2; ++c) {
               uint8_t expected[16], length = 0;
               if (!agx3_stage_fma(AGX_ISA_ENCODING_AGX3B, 2 * c, slot, src[c],
                                   AGX3_STAGE_FMA_UNIFORM, hints, hints, wait,
                                   expected, &length))
                  continue;
               for (unsigned offset = 0; offset + length <= binary.size;
                    ++offset) {
                  if (memcmp(bytes + offset, expected, length))
                     continue;
                  found[c]++;
                  if (c == 0) {
                     EXPECT_EQ(wait != 0, loaded);
                  }
                  publication_slot[c] = slot;
               }
            }
         }
      }
      EXPECT_EQ(found[0], 1u);
      EXPECT_EQ(found[1], 1u);
      EXPECT_EQ(publication_slot[0], publication_slot[1]);
      for (unsigned c = 0; c < 2; ++c) {
         for (unsigned slot = 1; slot <= 6; ++slot) {
            uint8_t copy[AGX3_STAGE_MAX_LENGTH], length = 0;
            ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B, 2 * c,
                                               outputs[c].value, false, slot, 0,
                                               copy, &length));
            for (unsigned offset = 0; offset + length <= binary.size; ++offset)
               EXPECT_NE(memcmp(bytes + offset, copy, length), 0);
         }
      }
      util_dynarray_fini(&binary);
      ralloc_free(mem);
   }
}
