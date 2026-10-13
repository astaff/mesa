/* SPDX-License-Identifier: MIT */
#include <cmath>
#include "util/half_float.h"
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "asahi/layout/layout.h"
#include "agx_nir.h"
}

static unsigned
lower_fma(unsigned slot, uint32_t bits, bool rte = true)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "constant FMA rounding");
   if (rte)
      b.shader->info.float_controls_execution_mode =
         FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32;
   b.fp_math_ctrl = nir_fp_preserve_sz_inf_nan;
   nir_def *coord = nir_load_frag_coord(&b);
   nir_def *src[3] = {nir_channel(&b, coord, 0), nir_channel(&b, coord, 1),
                      nir_channel(&b, coord, 2)};
   src[slot] = nir_imm_int(&b, bits);
   nir_def *value = nir_ffma(&b, src[0], src[1], src[2]);
   agx_test_store_local_pixel_agx(&b, nir_vec4(&b, value, value, value, value),
                                  nir_imm_intN_t(&b, 0xff, 16),
                                  nir_undef(&b, 2, 16), 0, 0xf,
                                  (enum pipe_format)AIL_ISA_FORMAT_I32);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_nir_lower_rte_fma(b.shader);
   nir_validate_shader(b.shader, "RTE FMA repair");
   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
   nir_index_blocks(impl);
   unsigned blocks = impl->num_blocks;
   ralloc_free(b.shader);
   return blocks;
}

TEST(RTEFMA, OmitsUnreachableRepairPaths)
{
   for (unsigned slot : {0u, 1u}) {
      for (uint32_t bits :
           {0u, 0x80000000u, 0x00000001u, 0x007fffffu, 0x3f800000u, 0x40000000u,
            0x7f000000u, 0xbf800000u, 0x7f800000u, 0x7fc00000u}) {
         SCOPED_TRACE(slot);
         SCOPED_TRACE(bits);
         EXPECT_EQ(lower_fma(slot, bits), 1u);
      }
   }
   for (uint32_t bits : {0x0d000000u, 0x0d7fffffu, 0x3f800000u, 0xbf800000u,
                         0x7f800000u, 0x7fc00000u}) {
      SCOPED_TRACE(bits);
      EXPECT_EQ(lower_fma(2, bits), 1u);
   }
}

TEST(RTEFMA, RetainsPossibleBoundaryRepairs)
{
   for (unsigned slot : {0u, 1u}) {
      for (uint32_t bits :
           {0x00800000u, 0x3f000000u, 0x3f7fffffu, 0x3f800001u, 0xbf000000u}) {
         SCOPED_TRACE(slot);
         SCOPED_TRACE(bits);
         EXPECT_GT(lower_fma(slot, bits), 1u);
      }
   }
   for (uint32_t bits : {0u, 0x80000000u, 0x00800000u, 0x0cffffffu}) {
      SCOPED_TRACE(bits);
      EXPECT_GT(lower_fma(2, bits), 1u);
   }
   EXPECT_EQ(lower_fma(2, 0x00800000u, false), 1u);
}

static bool
needs_boundary_repair(float x, float y, float z)
{
   auto flush = [](float f) {
      return (fui(f) & 0x7f800000u) == 0 ? std::copysign(0.0f, f) : f;
   };
   x = flush(x);
   y = flush(y);
   z = flush(z);
   double product = double(x) * double(y);
   double exact = product + double(z);
   double magnitude = std::fabs(exact);
   return magnitude >= std::ldexp(1.0, -126) - std::ldexp(1.0, -150) &&
          magnitude < std::ldexp(1.0, -126);
}

TEST(RTEFMA, ReferenceBoundaryExamples)
{
   EXPECT_TRUE(needs_boundary_repair(uif(0x00800000), uif(0x3f7fffff), 0));
   EXPECT_TRUE(needs_boundary_repair(uif(0x00ffffff), 0.5f, 0));
   EXPECT_TRUE(needs_boundary_repair(uif(0x00800000), -std::ldexp(1.0f, -24),
                                     uif(0x00800000)));
   EXPECT_EQ(fui(std::fma(uif(0x00800000), uif(0x3f7fffff), 0.0f)),
             0x00800000u);
}

TEST(RTEFMA, LargeConstantAddendsCannotReachBoundary)
{
   uint32_t state = 1;
   auto random = [&state]() {
      state = state * 1664525u + 1013904223u;
      return state;
   };
   for (unsigned exponent = 26; exponent < 255; ++exponent) {
      for (unsigned i = 0; i < 1024; ++i) {
         float z = uif((exponent << 23) | (random() & 0x807fffffu));
         float x = uif(((1 + random() % 254) << 23) | (random() & 0x807fffffu));
         float center = float(-double(z) / double(x));
         if (!std::isfinite(center))
            continue;
         float y = center;
         for (unsigned adjacent = 0; adjacent < 5; ++adjacent) {
            if (adjacent == 1)
               y = std::nextafter(center, -INFINITY);
            else if (adjacent == 3)
               y = std::nextafter(center, INFINITY);
            else if (adjacent)
               y = std::nextafter(y, adjacent < 3 ? -INFINITY : INFINITY);
            ASSERT_FALSE(needs_boundary_repair(x, y, z));
         }
      }
   }
}

TEST(RTEFMA, IntegralConstantMultipliersNeedNoRepair)
{
   for (unsigned slot : {0u, 1u}) {
      for (float multiplier : {3.0f, -3.0f, 15.0f, 255.0f, -65535.0f,
                               16777215.0f, uif(0x7f7fffff)}) {
         SCOPED_TRACE(slot);
         SCOPED_TRACE(multiplier);
         EXPECT_EQ(lower_fma(slot, fui(multiplier)), 1u);
      }
      for (float multiplier :
           {1.5f, -1.5f, 3.25f, std::nextafter(3.0f, INFINITY)}) {
         SCOPED_TRACE(multiplier);
         EXPECT_GT(lower_fma(slot, fui(multiplier)), 1u);
      }
   }
}

static unsigned
lower_integral_expression(nir_op producer, unsigned selection)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "integral FMA expression");
   b.shader->info.float_controls_execution_mode =
      FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32;
   b.fp_math_ctrl = nir_fp_preserve_sz_inf_nan;
   nir_def *coord = nir_load_frag_coord(&b);
   nir_def *x = nir_channel(&b, coord, 0);
   nir_def *z = nir_channel(&b, coord, 2);
   nir_def *condition = nir_flt_imm(&b, x, 0);
   nir_def *input = producer == nir_op_b2f32 || producer == nir_op_b2i32
                       ? condition
                       : nir_channel(&b, coord, 1);
   nir_def *factor = nir_build_alu(&b, producer, input, NULL, NULL, NULL);
   factor = nir_fabs(&b, nir_fneg(&b, factor));
   if (selection == 1 || selection == 2)
      factor = nir_bcsel(&b, condition, factor,
                         nir_imm_float(&b, selection == 1 ? 2.0f : 0.5f));
   else if (selection == 3)
      factor =
         nir_fadd(&b, nir_fmul_imm(&b, factor, 3.0f), nir_imm_float(&b, 4.0f));
   nir_def *value = nir_ffma(&b, x, factor, z);
   agx_test_store_local_pixel_agx(&b, nir_vec4(&b, value, value, value, value),
                                  nir_imm_intN_t(&b, 0xff, 16),
                                  nir_undef(&b, 2, 16), 0, 0xf,
                                  (enum pipe_format)AIL_ISA_FORMAT_I32);
   agx_nir_lower_rte_fma(b.shader);
   nir_validate_shader(b.shader, "integral expression FMA repair");
   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
   nir_index_blocks(impl);
   unsigned blocks = impl->num_blocks;
   ralloc_free(b.shader);
   return blocks;
}

TEST(RTEFMA, IntegralProducerExpressionsNeedNoRepair)
{
   for (nir_op op : {nir_op_i2f32, nir_op_u2f32, nir_op_b2f32, nir_op_ftrunc,
                     nir_op_ffloor, nir_op_fceil, nir_op_fround_even}) {
      for (unsigned selection = 0; selection < 4; ++selection) {
         SCOPED_TRACE(op);
         SCOPED_TRACE(selection);
         if (selection == 2)
            EXPECT_GT(lower_integral_expression(op, selection), 1u);
         else
            EXPECT_EQ(lower_integral_expression(op, selection), 1u);
      }
   }
   for (nir_op op : {nir_op_b2i32, nir_op_fsqrt, nir_op_frcp})
      EXPECT_GT(lower_integral_expression(op, 0), 1u);
}

TEST(RTEFMA, IntegralMultipliersCannotReachReferenceBoundary)
{
   uint32_t state = 123;
   auto random = [&state]() {
      state = state * 1664525u + 1013904223u;
      return state;
   };
   for (unsigned i = 0; i < 250000; ++i) {
      float x = uif(((1 + random() % 254) << 23) | (random() & 0x807fffffu));
      float y = float(int32_t(random()));
      if ((i & 3) == 0)
         y = (i & 4) ? -3.0f : 3.0f;
      float center = float(-double(x) * double(y));
      if (!std::isfinite(center))
         continue;
      for (float z : {center, std::nextafter(center, -INFINITY),
                      std::nextafter(center, INFINITY)})
         ASSERT_FALSE(needs_boundary_repair(x, y, z));
   }
}

static unsigned
lower_large_addend_expression(unsigned form)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "bounded FMA addend");
   b.shader->info.float_controls_execution_mode =
      FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32;
   b.fp_math_ctrl = nir_fp_preserve_sz_inf_nan;
   nir_def *coord = nir_load_frag_coord(&b);
   nir_def *q = nir_channel(&b, coord, 2);
   nir_def *z;
   switch (form) {
   case 0:
      z = nir_fadd_imm(&b, nir_fabs(&b, q), 1.0f);
      break;
   case 1:
      z = nir_fmax(&b, q, nir_imm_float(&b, 1.0f));
      break;
   case 2:
      z = nir_fmin(&b, q, nir_imm_float(&b, -1.0f));
      break;
   case 3:
      z = nir_bcsel(&b, nir_flt_imm(&b, q, 0), nir_imm_float(&b, -1.0f),
                    nir_fadd_imm(&b, nir_fabs(&b, q), 2.0f));
      break;
   case 4:
   case 5:
      z = nir_bcsel(&b, nir_flt_imm(&b, q, 0), nir_imm_float(&b, 2.0f),
                    nir_imm_float(&b, form == 4 ? 0.0f : 0.5f));
      break;
   case 6:
      z = nir_fmin(&b, nir_fabs(&b, q), nir_imm_float(&b, 1.0f));
      break;
   case 7:
      z = nir_fadd_imm(&b, nir_fabs(&b, q), 0.5f);
      break;
   case 8:
      z = nir_fadd_imm(&b, nir_fabs(&b, q), uif(0x0d000000));
      break;
   case 9:
      z = nir_fadd_imm(&b, nir_fneg(&b, nir_fabs(&b, q)), -uif(0x0d000000));
      break;
   case 10:
      z = nir_fadd_imm(&b, nir_fabs(&b, q), uif(0x0c800000));
      break;
   case 11:
      z = nir_fmax(&b, q, nir_imm_float(&b, uif(0x0d000000)));
      break;
   case 12:
      z = nir_fmin(&b, q, nir_imm_float(&b, -uif(0x0d000000)));
      break;
   case 13:
      z = nir_fadd_imm(&b, q, 1.0f);
      break;
   default:
      z = nir_fadd_imm(&b, nir_fabs(&b, q), uif(1));
      break;
   }
   nir_def *value =
      nir_ffma(&b, nir_channel(&b, coord, 0), nir_channel(&b, coord, 1), z);
   agx_test_store_local_pixel_agx(&b, nir_vec4(&b, value, value, value, value),
                                  nir_imm_intN_t(&b, 0xff, 16),
                                  nir_undef(&b, 2, 16), 0, 0xf,
                                  (enum pipe_format)AIL_ISA_FORMAT_I32);
   nir_lower_alu(b.shader);
   agx_nir_lower_fminmax(b.shader);
   agx_nir_lower_rte_fma(b.shader);
   nir_validate_shader(b.shader, "bounded addend FMA repair");
   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
   nir_index_blocks(impl);
   unsigned blocks = impl->num_blocks;
   ralloc_free(b.shader);
   return blocks;
}

TEST(RTEFMA, DynamicLargeAddendsNeedNoRepair)
{
   for (unsigned form : {0u, 1u, 2u, 3u, 5u, 7u, 8u, 9u, 11u, 12u}) {
      SCOPED_TRACE(form);
      EXPECT_EQ(lower_large_addend_expression(form), 1u);
   }
   for (unsigned form : {4u, 6u, 10u, 13u, 14u}) {
      SCOPED_TRACE(form);
      EXPECT_GT(lower_large_addend_expression(form), 1u);
   }
}

TEST(RTEFMA, BoundedAddendExpressionsCannotReachReferenceBoundary)
{
   uint32_t state = 987;
   auto random = [&state]() {
      state = state * 1664525u + 1013904223u;
      return state;
   };
   for (float bias : {1.0f, 0.5f, uif(0x0d000000)}) {
      for (unsigned i = 0; i < 32768; ++i) {
         float q = i < 256 ? 0.0f : uif(random());
         if (!std::isfinite(q))
            continue;
         for (float z : {std::fabs(q) + bias, -(std::fabs(q) + bias),
                         std::fmax(q, bias), std::fmin(q, -bias)}) {
            if (!std::isfinite(z))
               continue;
            float x =
               uif(((1 + random() % 254) << 23) | (random() & 0x807fffffu));
            float center = float(-double(z) / double(x));
            if (!std::isfinite(center))
               continue;
            for (float y : {center, std::nextafter(center, -INFINITY),
                            std::nextafter(center, INFINITY)})
               ASSERT_FALSE(needs_boundary_repair(x, y, z));
         }
      }
   }
}

static unsigned
lower_grid_expression(unsigned mode, uint32_t factor_bits,
                      unsigned component = 0)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "fixed-grid FMA expression");
   b.shader->info.float_controls_execution_mode =
      FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32;
   b.fp_math_ctrl = nir_fp_preserve_sz_inf_nan;
   nir_def *raw = nir_load_frag_coord(&b);
   nir_def *a = agx_test_load_preamble(&b, 1, 16, 0);
   nir_def *c = agx_test_load_preamble(&b, 1, 16, 1);
   nir_def *half_a = nir_f2f32(&b, a);
   nir_def *half_c = nir_f2f32(&b, c);
   nir_def *x = half_a;
   nir_def *y = mode == 0 ? half_c : nir_imm_int(&b, factor_bits);
   if (mode == 1)
      y = nir_channel(&b, raw, 1);
   if (mode == 3)
      x = nir_fadd(&b, half_a, half_c);
   if (mode == 4)
      x = nir_fmul_imm(&b, half_a, 1.0f / 3.0f);
   if (mode == 5) {
      nir_def *condition = nir_flt_imm(&b, nir_channel(&b, raw, 0), 0.0);
      x = nir_bcsel(&b, condition, half_a, nir_channel(&b, raw, 1));
   }
   if (mode == 6) {
      nir_def *condition = nir_flt_imm(&b, nir_channel(&b, raw, 0), 0.0);
      x = nir_bcsel(&b, condition, nir_fneg(&b, half_a), nir_fabs(&b, half_c));
   }
   if (mode == 7)
      x = nir_channel(&b, nir_vec2(&b, half_a, nir_channel(&b, raw, 1)),
                      component);
   if (mode == 8 || mode == 9) {
      nir_def *condition = nir_flt_imm(&b, nir_channel(&b, raw, 0), 0.0);
      nir_push_if(&b, condition);
      nir_def *one = half_a;
      nir_push_else(&b, NULL);
      nir_def *two = mode == 8 ? half_c : nir_channel(&b, raw, 1);
      nir_pop_if(&b, NULL);
      x = nir_if_phi(&b, one, two);
   }
   if (mode == 10)
      x = nir_fmul_imm(&b, nir_channel(&b, raw, 0), 2.0f);
   if (mode == 11) {
      x = nir_fmul(&b, nir_u2f32(&b, nir_channel(&b, raw, 0)),
                   nir_imm_int(&b, 0x0d800000));
   }
   if (mode == 12) {
      for (unsigned i = 0; i < component; ++i)
         x = nir_fneg(&b, x);
   }
   nir_def *value = nir_ffma(&b, x, y, nir_channel(&b, raw, 2));
   agx_test_store_local_pixel_agx(&b, nir_vec4(&b, value, value, value, value),
                                  nir_imm_intN_t(&b, 0xff, 16),
                                  nir_undef(&b, 2, 16), 0, 0xf,
                                  (enum pipe_format)AIL_ISA_FORMAT_I32);
   agx_nir_lower_rte_fma(b.shader);
   nir_validate_shader(b.shader, "grid expression FMA repair");
   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
   nir_index_blocks(impl);
   unsigned blocks = impl->num_blocks;
   ralloc_free(b.shader);
   return blocks;
}

TEST(RTEFMA, HalfPrecisionProductsLieOnSafeGrids)
{
   EXPECT_EQ(lower_grid_expression(0, 0), 1u);
   EXPECT_EQ(lower_grid_expression(2, 0x3f000000), 1u);
   EXPECT_EQ(lower_grid_expression(2, 0x01000000), 1u);
   EXPECT_EQ(lower_grid_expression(3, 0x01000000), 1u);
   EXPECT_EQ(lower_grid_expression(4, 0x0d800000), 1u);
   EXPECT_EQ(lower_grid_expression(6, 0x3f000000), 1u);
}

TEST(RTEFMA, GridProofRetainsUnknownAndFineProducts)
{
   EXPECT_GT(lower_grid_expression(1, 0), 1u);
   EXPECT_GT(lower_grid_expression(2, 0x00800000), 1u);
   EXPECT_GT(lower_grid_expression(3, 0x00800000), 1u);
   EXPECT_GT(lower_grid_expression(4, 0x0d000000), 1u);
   EXPECT_GT(lower_grid_expression(5, 0x3f000000), 1u);
}

TEST(RTEFMA, GridCacheKeepsComponentsSeparate)
{
   EXPECT_EQ(lower_grid_expression(7, 0x3f000000, 0), 1u);
   EXPECT_GT(lower_grid_expression(7, 0x3f000000, 1), 1u);
}

TEST(RTEFMA, GridPhiRequiresEveryIncomingPath)
{
   EXPECT_EQ(lower_grid_expression(8, 0x3f000000), 4u);
   EXPECT_GT(lower_grid_expression(9, 0x3f000000), 4u);
}

TEST(RTEFMA, HalfGridNumericalBoundaryAndCounterexample)
{
   const float min_normal = uif(0x00800000);
   EXPECT_TRUE(needs_boundary_repair(_mesa_half_to_float(0x8001), min_normal,
                                     min_normal));
   EXPECT_FALSE(needs_boundary_repair(_mesa_half_to_float(0x8001),
                                      uif(0x01000000), min_normal));
   uint32_t state = 121;
   auto random = [&state]() {
      state = state * 1664525u + 1013904223u;
      return state;
   };
   for (unsigned bits = 0; bits <= 0xffff; ++bits) {
      float x = _mesa_half_to_float(bits);
      float y = _mesa_half_to_float(random() & 0xffff);
      if (!std::isfinite(x) || !std::isfinite(y))
         continue;
      float cancel = float(-double(x) * double(y));
      for (float z : {0.0f, min_normal, -min_normal, cancel,
                      std::nextafter(cancel, INFINITY),
                      std::nextafter(cancel, -INFINITY)})
         ASSERT_FALSE(needs_boundary_repair(x, y, z)) << bits;
      for (float z :
           {min_normal, -min_normal, uif(0x00800001), -uif(0x00800001)})
         ASSERT_FALSE(needs_boundary_repair(x, uif(0x01000000), z)) << bits;
   }
}

TEST(RTEFMA, FloatScalingAndIntegerExpressionsKeepProductGrids)
{
   EXPECT_EQ(lower_grid_expression(10, 0x3f000000), 1u);
   EXPECT_GT(lower_grid_expression(10, 0x3e800000), 1u);
   EXPECT_EQ(lower_grid_expression(11, 0x27000000), 1u);
   EXPECT_GT(lower_grid_expression(11, 0x26800000), 1u);
}

TEST(RTEFMA, GridDepthLimitRetainsConservativeRepair)
{
   EXPECT_EQ(lower_grid_expression(12, 0x3f000000, 32), 1u);
   EXPECT_GT(lower_grid_expression(12, 0x3f000000, 128), 1u);
}

TEST(RTEFMA, CyclicPhiKeepsConservativeGrid)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "cyclic fixed-grid FMA");
   b.shader->info.float_controls_execution_mode =
      FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32;
   b.fp_math_ctrl = nir_fp_preserve_sz_inf_nan;
   nir_def *raw = nir_load_frag_coord(&b);
   nir_def *initial = nir_f2f32(&b, agx_test_load_preamble(&b, 1, 16, 0));
   nir_variable *state =
      nir_local_variable_create(b.impl, glsl_float_type(), "state");
   nir_store_var(&b, state, initial, 1);
   nir_push_loop(&b);
   nir_def *current = nir_load_var(&b, state);
   nir_def *value =
      nir_ffma(&b, current, nir_imm_float(&b, 0.5), nir_channel(&b, raw, 2));
   agx_test_store_local_pixel_agx(&b, nir_vec4(&b, value, value, value, value),
                                  nir_imm_intN_t(&b, 0xff, 16),
                                  nir_undef(&b, 2, 16), 0, 0xf,
                                  (enum pipe_format)AIL_ISA_FORMAT_I32);
   nir_push_if(&b, nir_flt_imm(&b, current, 0));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);
   nir_store_var(&b, state, nir_fadd_imm(&b, current, 0.25), 1);
   nir_pop_loop(&b, NULL);
   nir_lower_vars_to_ssa(b.shader);
   nir_index_blocks(b.impl);
   unsigned before = b.impl->num_blocks;
   EXPECT_TRUE(agx_nir_lower_rte_fma(b.shader));
   nir_validate_shader(b.shader, "cyclic grid repair");
   nir_index_blocks(b.impl);
   EXPECT_GT(b.impl->num_blocks, before);
   ralloc_free(b.shader);
}
