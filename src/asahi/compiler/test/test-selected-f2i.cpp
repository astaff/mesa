#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_constant_expressions.h"
extern "C" {
#include "agx_nir.h"
}

static bool
valid_signed_conversion(uint32_t word)
{
   uint32_t magnitude = word & 0x7fffffff;
   return magnitude < 0x4f000000 ||
          (magnitude == 0x4f000000 && (word & 0x80000000));
}

static uint32_t
truncate_signed_bits(uint32_t word)
{
   uint32_t exponent = (word >> 23) & 255;
   if (exponent < 127)
      return 0;
   uint64_t significand = (word & 0x7fffff) | 0x800000;
   unsigned power = exponent - 127;
   uint64_t magnitude = power >= 23 ? significand << (power - 23)
                                    : significand >> (23 - power);
   return word >> 31 ? 0u - uint32_t(magnitude) : uint32_t(magnitude);
}

static nir_const_value
evaluate_selected_conversion(nir_scalar scalar, nir_def *a, nir_def *b,
                             nir_def *choice, uint32_t x, uint32_t y,
                             bool selected, unsigned mode)
{
   scalar = nir_scalar_chase_movs(scalar);
   if (scalar.def == a || scalar.def == b || scalar.def == choice)
      return nir_const_value_for_uint(scalar.def == a ? x : scalar.def == b ? y
                                                                      : selected,
                                       scalar.def->bit_size);
   if (nir_scalar_is_const(scalar))
      return nir_const_value_for_uint(nir_scalar_as_uint(scalar),
                                       scalar.def->bit_size);
   if (!nir_scalar_is_alu(scalar)) {
      ADD_FAILURE() << "Unexpected conversion instruction";
      return {};
   }
   nir_alu_instr *alu = nir_scalar_as_alu(scalar);
   const nir_op_info *info = &nir_op_infos[alu->op];
   nir_const_value values[NIR_ALU_MAX_INPUTS] = {};
   nir_const_value *sources[NIR_ALU_MAX_INPUTS] = {};
   unsigned bits = nir_alu_type_get_type_size(info->output_type) == 0
                   ? alu->def.bit_size : 0;
   for (unsigned i = 0; i < info->num_inputs; ++i) {
      nir_scalar input = nir_scalar_chase_alu_src(scalar, i);
      values[i] = evaluate_selected_conversion(input, a, b, choice, x, y,
                                               selected, mode);
      sources[i] = &values[i];
      if (!bits && nir_alu_type_get_type_size(info->input_types[i]) == 0)
         bits = input.def->bit_size;
   }
   nir_const_value result = {};
   uint16_t poison = 0;
   nir_eval_const_opcode(alu->op, &result, &poison, 1, bits ? bits : 32,
                         sources, mode);
   EXPECT_FALSE(poison);
   return result;
}

TEST(SelectedF2I, PreservesTruncationForDefinedInputsAndControls)
{
   const std::array<uint32_t, 24> edges = {
      0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
      0x00800000, 0x80800000, 0x3f000000, 0xbf000000,
      0x3f7fffff, 0xbf7fffff, 0x3f800000, 0xbf800000,
      0x3fc00000, 0xbfc00000, 0x40200000, 0xc0200000,
      0x4b800000, 0xcb800000, 0x4effffff, 0xceffffff,
      0xcf000000, 0x3f800001};
   for (unsigned mode : {0u, unsigned(FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32),
                          unsigned(FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP32),
                          unsigned(FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32)}) {
      for (unsigned flags : {unsigned(nir_fp_fast_math), unsigned(nir_fp_exact)}) {
         SCOPED_TRACE(mode);
         SCOPED_TRACE(flags);
         nir_builder nb = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "selected FP32 integer conversion");
         nb.shader->info.float_controls_execution_mode = mode;
         nb.fp_math_ctrl = flags;
         nir_def *a = agx_test_load_preamble(&nb, 1, 32, 0);
         nir_def *b = agx_test_load_preamble(&nb, 1, 32, 4);
         nir_def *choice = agx_test_load_preamble(&nb, 1, 32, 8);
         nir_def *keep = nir_mov(&nb, nir_bcsel(&nb, nir_ine_imm(&nb, choice, 0),
                                   nir_f2i32(&nb, a), nir_f2i32(&nb, b)));
         ASSERT_TRUE(agx_nir_fuse_algebraic_late(nb.shader));
         nir_scalar result = nir_scalar_chase_alu_src(nir_get_scalar(keep, 0), 0);
         ASSERT_EQ(nir_scalar_alu_op(result), nir_op_f2i32);
         EXPECT_EQ(nir_scalar_as_alu(result)->fp_math_ctrl, flags);
         EXPECT_EQ(nir_scalar_alu_op(nir_scalar_chase_alu_src(result, 0)), nir_op_bcsel);
         auto check = [&](uint32_t x, uint32_t y) {
            ASSERT_TRUE(valid_signed_conversion(x));
            ASSERT_TRUE(valid_signed_conversion(y));
            for (bool selected : {false, true}) {
               nir_const_value actual = evaluate_selected_conversion(
                  result, a, b, choice, x, y, selected, mode);
               EXPECT_EQ(nir_const_value_as_uint(actual, 32),
                         truncate_signed_bits(selected ? x : y))
                  << x << " " << y << " " << selected;
            }
         };
         for (uint32_t x : edges)
            for (uint32_t y : edges)
               check(x, y);
         uint32_t state = 0x83274615;
         auto next = [&]() {
            do {
               state ^= state << 13;
               state ^= state >> 17;
               state ^= state << 5;
            } while (!valid_signed_conversion(state));
            return state;
         };
         for (unsigned i = 0; i < 10000; ++i) {
            uint32_t x = next(), y = next();
            check(x, y);
         }
         nir_validate_shader(nb.shader, "selected FP32 integer conversion");
         ralloc_free(nb.shader);
      }
   }
}

TEST(SelectedF2I, CollapsesThreeWaySelection)
{
   nir_builder nb = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "three coordinate alternatives");
   nir_def *a = agx_test_load_preamble(&nb, 1, 32, 0);
   nir_def *b = agx_test_load_preamble(&nb, 1, 32, 4);
   nir_def *c = agx_test_load_preamble(&nb, 1, 32, 8);
   nir_def *choice = agx_test_load_preamble(&nb, 1, 32, 12);
   nir_def *inner = nir_bcsel(&nb, nir_ine_imm(&nb, choice, 0),
                             nir_f2i32(&nb, a), nir_f2i32(&nb, b));
   nir_def *out = nir_bcsel(&nb, nir_ine_imm(&nb, choice, 1),
                           nir_f2i32(&nb, c), inner);
   nir_def *keep = nir_mov(&nb, out);
   unsigned rounds = 0;
   while (agx_nir_fuse_algebraic_late(nb.shader))
      ASSERT_LT(++rounds, 4u);
   EXPECT_GT(rounds, 0u);
   nir_scalar result = nir_scalar_chase_alu_src(nir_get_scalar(keep, 0), 0);
   ASSERT_EQ(nir_scalar_alu_op(result), nir_op_f2i32);
   nir_scalar select = nir_scalar_chase_alu_src(result, 0);
   ASSERT_EQ(nir_scalar_alu_op(select), nir_op_bcsel);
   EXPECT_EQ(nir_scalar_alu_op(nir_scalar_chase_alu_src(select, 2)), nir_op_bcsel);
   nir_validate_shader(nb.shader, "three coordinate alternatives");
   ralloc_free(nb.shader);
}

TEST(SelectedF2I, LeavesSharedMixedAndOtherWidthsUnchanged)
{
   for (unsigned control = 0; control < 7; ++control) {
      SCOPED_TRACE(control);
      nir_builder nb = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "unrelated conversions");
      nb.fp_math_ctrl = nir_fp_fast_math;
      unsigned components = control == 6 ? 2 : 1;
      nir_def *a = agx_test_load_preamble(&nb, components, control == 4 ? 16 : 32, 0);
      nir_def *b = agx_test_load_preamble(&nb, components, 32, 4);
      nir_def *choice = agx_test_load_preamble(&nb, 1, 32, 8);
      nir_def *left = control == 3 ? nir_f2i16(&nb, a) : nir_f2i32(&nb, a);
      nir_def *right = control == 2 ? nir_f2i32_rtne(&nb, b)
                     : control == 3 ? nir_f2i16(&nb, b) : nir_f2i32(&nb, b);
      if (control == 0)
         agx_test_store_preamble(&nb, left, 20);
      if (control == 1)
         agx_test_store_preamble(&nb, right, 20);
      if (control == 5)
         nir_scalar_as_alu(nir_get_scalar(left, 0))->fp_math_ctrl = nir_fp_exact;
      nir_mov(&nb, nir_bcsel(&nb, nir_ine_imm(&nb, choice, 0), left, right));
      EXPECT_FALSE(agx_nir_fuse_algebraic_late(nb.shader));
      nir_validate_shader(nb.shader, "unrelated conversions");
      ralloc_free(nb.shader);
   }
}
