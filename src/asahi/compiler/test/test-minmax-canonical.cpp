#include <cstdint>
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_constant_expressions.h"

static nir_const_value
evaluate_minmax(nir_scalar value, nir_def *a, nir_def *b, uint64_t x,
                uint64_t y, unsigned mode)
{
   if (value.def == a || value.def == b)
      return nir_const_value_for_uint(value.def == a ? x : y,
                                      value.def->bit_size);
   if (nir_scalar_is_const(value))
      return nir_const_value_for_uint(nir_scalar_as_uint(value),
                                      value.def->bit_size);
   if (!nir_scalar_is_alu(value)) {
      ADD_FAILURE() << "Unexpected instruction";
      return {};
   }
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(value.def));
   const nir_op_info *info = &nir_op_infos[alu->op];
   nir_const_value sources[NIR_ALU_MAX_INPUTS] = {};
   nir_const_value *pointers[NIR_ALU_MAX_INPUTS] = {};
   unsigned bits = nir_alu_type_get_type_size(info->output_type) == 0
                   ? alu->def.bit_size : 0;
   for (unsigned i = 0; i < info->num_inputs; ++i) {
      nir_scalar src = nir_scalar_chase_alu_src(value, i);
      sources[i] = evaluate_minmax(src, a, b, x, y, mode);
      pointers[i] = &sources[i];
      if (!bits && nir_alu_type_get_type_size(info->input_types[i]) == 0)
         bits = src.def->bit_size;
   }
   nir_const_value result = {};
   nir_eval_const_opcode(alu->op, &result, nullptr, 1, bits ? bits : 32,
                         pointers, mode);
   return result;
}

TEST(MinmaxCanonical, PreservesSpecialValuesAcrossWidthsAndFlushModes)
{
   for (unsigned bits : {16u, 32u, 64u}) {
      const unsigned fraction = bits == 16 ? 10 : bits == 32 ? 23 : 52;
      const uint64_t sign = uint64_t(1) << (bits - 1);
      const uint64_t mask = sign | (sign - 1);
      const uint64_t mantissa = (uint64_t(1) << fraction) - 1;
      const uint64_t exponent = (sign - 1) & ~mantissa;
      const unsigned ftz = bits == 16
         ? FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP16 : bits == 32
         ? FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32
         : FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP64;
      for (bool flush : {false, true}) {
         for (bool maximum : {false, true}) {
            nir_builder nb = nir_builder_init_simple_shader(
               MESA_SHADER_COMPUTE, &agx_nir_options, "selected minmax result");
            nb.shader->info.float_controls_execution_mode = flush ? ftz : 0;
            nb.fp_math_ctrl = nir_fp_preserve_sz_inf_nan;
            nir_def *a = agx_test_load_preamble(&nb, 1, bits, 0);
            nir_def *b = agx_test_load_preamble(&nb, 1, bits, 4);
            nir_def *out = maximum ? nir_fmax(&nb, a, b) : nir_fmin(&nb, a, b);
            nir_def *keep = nir_mov(&nb, out);
            ASSERT_TRUE(nir_lower_alu(nb.shader));
            nir_scalar result = nir_scalar_chase_alu_src(nir_get_scalar(keep, 0), 0);
            auto canonical = [&](uint64_t word) {
               return flush && !(word & exponent) ? word & sign : word;
            };
            auto nan = [&](uint64_t word) {
               return (word & exponent) == exponent && (word & mantissa);
            };
            auto check = [&](uint64_t x, uint64_t y) {
               x &= mask; y &= mask;
               const uint64_t quiet = uint64_t(1) << (fraction - 1);
               if (bits == 64 && ((nan(x) && !(x & quiet)) ||
                                  (nan(y) && !(y & quiet))))
                  return;
               uint64_t cx = canonical(x), cy = canonical(y);
               nir_const_value actual = evaluate_minmax(
                  result, a, b, x, y, flush ? ftz : 0);
               uint64_t got = nir_const_value_as_uint(actual, bits);
               if (nan(cx) && nan(cy)) {
                  EXPECT_TRUE(nan(got));
                  return;
               }
               uint64_t expected;
               if (nan(cx)) expected = cy;
               else if (nan(cy)) expected = cx;
               else if (!(cx & ~sign) && !(cy & ~sign))
                  expected = maximum ? cx & cy : cx | cy;
               else {
                  uint64_t ox = cx & sign ? (~cx & mask) : cx | sign;
                  uint64_t oy = cy & sign ? (~cy & mask) : cy | sign;
                  expected = (maximum ? ox > oy : ox < oy) ? cx : cy;
               }
               EXPECT_EQ(got, expected) << bits << " " << flush << " "
                  << maximum << " " << x << " " << y;
            };
            const uint64_t edges[] = {0, sign, 1, sign | 1, mantissa,
               sign | mantissa, mantissa + 1, sign | (mantissa + 1),
               exponent - 1, sign | (exponent - 1), exponent, sign | exponent,
               exponent | 1, sign | exponent | 1, exponent | mantissa,
               sign | exponent | mantissa};
            for (uint64_t x : edges)
               for (uint64_t y : edges) check(x, y);
            uint64_t random = 0x987654321fedcbaULL;
            for (unsigned i = 0; i < 10000; ++i) {
               random ^= random << 13; random ^= random >> 7;
               random ^= random << 17;
               uint64_t x = random;
               random ^= random << 13; random ^= random >> 7;
               random ^= random << 17;
               check(x, random);
            }
            nir_validate_shader(nb.shader, "selected minmax result");
            ralloc_free(nb.shader);
         }
      }
   }
}
