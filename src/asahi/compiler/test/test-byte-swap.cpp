#include <cstdint>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
}

static uint32_t
evaluate_swap(nir_scalar value, nir_def *input, uint32_t word)
{
   value = nir_scalar_chase_movs(value);
   if (value.def == input)
      return word;
   if (nir_scalar_is_const(value))
      return nir_scalar_as_uint(value);
   if (!nir_scalar_is_alu(value)) {
      ADD_FAILURE() << "Unexpected instruction";
      return 0;
   }
   auto src = [&](unsigned i) {
      return evaluate_swap(nir_scalar_chase_alu_src(value, i), input, word);
   };
   switch (nir_scalar_alu_op(value)) {
   case nir_op_ior: return src(0) | src(1);
   case nir_op_iand: return src(0) & src(1);
   case nir_op_ishl: return src(0) << (src(1) & 31);
   case nir_op_ushr: return src(0) >> (src(1) & 31);
   case nir_op_extr_agx: {
      uint64_t pair = (uint64_t(src(1)) << 32) | src(0);
      uint32_t shift = src(2) & 127;
      uint32_t result = shift >= 64 ? 0 : uint32_t(pair >> shift);
      return src(3) ? result & ((1u << src(3)) - 1) : result;
   }
   default: ADD_FAILURE() << "Unexpected opcode"; return 0;
   }
}

static nir_def *
swap_expression(nir_builder *b, nir_def *input, bool share, bool wrong_mask)
{
   nir_def *shift = nir_ushr_imm(b, input, 8);
   if (share)
      nir_store_agx(b, shift, nir_imm_int64(b, 0x1000), nir_imm_int(b, 0));
   nir_def *middle = nir_iand_imm(b, shift, wrong_mask ? 0xff01 : 0xff00);
   nir_def *high = nir_ior(b, nir_ishl_imm(b, input, 24),
                            nir_iand_imm(b, nir_ishl_imm(b, input, 8), 0xff0000));
   return nir_ior(b, nir_ushr_imm(b, input, 24), nir_ior(b, middle, high));
}

TEST(ByteSwap, ExactAllByteValuesAndWordEdges)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "word byte order");
   nir_def *input = agx_test_load_preamble(&b, 1, 32, 0);
   nir_def *keep = nir_mov(&b, swap_expression(&b, input, false, false));
   ASSERT_TRUE(agx_nir_fuse_algebraic_late(b.shader));
   nir_scalar output = nir_scalar_chase_alu_src(nir_get_scalar(keep, 0), 0);
   auto check = [&](uint32_t word) {
      uint32_t expected = ((word & 255) << 24) | ((word & 0xff00) << 8) |
                          ((word >> 8) & 0xff00) | (word >> 24);
      EXPECT_EQ(evaluate_swap(output, input, word), expected) << word;
   };
   for (uint32_t byte = 0; byte < 256; ++byte)
      for (unsigned shift : {0u, 8u, 16u, 24u}) {
         check(byte << shift);
         check(~(byte << shift));
      }
   for (uint32_t word : {0u, ~0u, 0x12345678u, 0x80000000u, 0x7fffffffu})
      check(word);
   uint32_t random = 0x98762341;
   for (unsigned i = 0; i < 10000; ++i) {
      random ^= random << 13; random ^= random >> 17; random ^= random << 5;
      check(random);
   }
   nir_validate_shader(b.shader, "word byte order");
   ralloc_free(b.shader);
}

TEST(ByteSwap, LeavesSharedAndDifferentMasksUnchanged)
{
   for (unsigned control = 0; control < 2; ++control) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "unrelated byte operations");
      nir_def *input = agx_test_load_preamble(&b, 1, 32, 0);
      nir_mov(&b, swap_expression(&b, input, control == 0, control == 1));
      EXPECT_FALSE(agx_nir_fuse_algebraic_late(b.shader));
      ralloc_free(b.shader);
   }
}
