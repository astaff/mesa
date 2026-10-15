#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
}

static uint32_t
evaluate_bounds(nir_scalar scalar, nir_def *offset, nir_def *bound,
                uint32_t offset_value, uint32_t bound_value)
{
   scalar = nir_scalar_chase_movs(scalar);
   if (scalar.def == offset)
      return offset_value;
   if (scalar.def == bound)
      return bound_value;
   if (nir_scalar_is_const(scalar))
      return nir_scalar_as_uint(scalar);
   if (!nir_scalar_is_alu(scalar)) {
      ADD_FAILURE() << "Unexpected bounds instruction";
      return 0;
   }
   auto source = [&](unsigned i) {
      return evaluate_bounds(nir_scalar_chase_alu_src(scalar, i), offset, bound,
                             offset_value, bound_value);
   };
   switch (nir_scalar_alu_op(scalar)) {
   case nir_op_iadd:
      return source(0) + source(1);
   case nir_op_iand:
      return source(0) & source(1);
   case nir_op_ushr:
      return source(0) >> (source(1) & 31);
   case nir_op_ult:
      return source(0) < source(1);
   default:
      ADD_FAILURE() << "Unexpected bounds opcode";
      return 0;
   }
}

static void
indexed_guard_store(nir_builder *b, nir_def *predicate, nir_def *offset,
                    unsigned shift, uint32_t delta)
{
   nir_push_if(b, predicate);
   nir_def *index = delta ? nir_iadd_imm(b, offset, delta) : offset;
   index = nir_ushr_imm(b, index, shift);
   nir_intrinsic_instr *store = nir_store_agx(
      b, nir_imm_intN_t(b, 0x1234, 8u << shift), nir_imm_int64(b, 0x1000), index);
   nir_intrinsic_set_format(store, shift == 1 ? PIPE_FORMAT_R16_UINT
                                            : PIPE_FORMAT_R32_UINT);
   nir_pop_if(b, NULL);
}

TEST(AlignedBounds, PreservesPartialRangesAndWrappingOffsets)
{
   const std::array<uint32_t, 22> edges = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17,
      0x7ffffffe, 0x7fffffff, 0x80000000, 0x80000001,
      0xfffffff0, 0xfffffffc, 0xfffffffd, 0xfffffffe, 0xffffffff};
   for (unsigned shift : {1u, 2u}) {
      uint32_t size = 1u << shift;
      for (uint32_t component = 0; component < 4; ++component) {
         SCOPED_TRACE(shift);
         SCOPED_TRACE(component);
         uint32_t delta = component * size;
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "aligned store bounds");
         nir_def *offset = agx_test_load_preamble(&b, 1, 32, 0);
         nir_def *bound = agx_test_load_preamble(&b, 1, 32, 4);
         nir_def *last = nir_iadd_imm(
            &b, nir_iand_imm(&b, offset, -size), delta + size - 1);
         nir_def *predicate = nir_ult(&b, last, bound);
         nir_def *keep = nir_b2i32(&b, predicate);
         indexed_guard_store(&b, predicate, offset, shift, delta);
         EXPECT_TRUE(agx_nir_fuse_algebraic_late(b.shader));
         nir_scalar result = nir_scalar_chase_alu_src(nir_get_scalar(keep, 0), 0);
         ASSERT_EQ(nir_scalar_alu_op(result), nir_op_ult);
         for (unsigned i = 0; i < 2; ++i)
            EXPECT_EQ(nir_scalar_alu_op(nir_scalar_chase_alu_src(result, i)),
                      nir_op_ushr);
         auto check = [&](uint32_t x, uint32_t limit) {
            uint32_t last_byte = (x & -size) + delta + size - 1;
            ASSERT_EQ(evaluate_bounds(result, offset, bound, x, limit),
                      uint32_t(last_byte < limit))
               << x << " " << limit;
         };
         for (uint32_t x : edges)
            for (uint32_t limit : edges)
               check(x, limit);
         for (uint32_t x = 0; x < 256; ++x)
            for (uint32_t limit = 0; limit < 32; ++limit)
               check(x, limit);
         uint32_t state = 0x68492231;
         auto random = [&]() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return state;
         };
         for (unsigned i = 0; i < 4096; ++i) {
            uint32_t x = random();
            uint32_t limit = random();
            check(x, limit);
            uint32_t last_byte = (x & -size) + delta + size - 1;
            check(x, last_byte);
            check(x, last_byte + 1);
         }
         nir_validate_shader(b.shader, "aligned store bounds");
         ralloc_free(b.shader);
      }
   }
}

TEST(AlignedBounds, RejectsIncompleteEndMasks)
{
   for (unsigned shift : {1u, 2u}) {
      uint32_t size = 1u << shift;
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "unaligned store bounds");
      nir_def *offset = agx_test_load_preamble(&b, 1, 32, 0);
      nir_def *bound = agx_test_load_preamble(&b, 1, 32, 4);
      nir_def *last = nir_iadd_imm(&b, nir_iand_imm(&b, offset, -size), size - 2);
      nir_def *predicate = nir_ult(&b, last, bound);
      nir_b2i32(&b, predicate);
      indexed_guard_store(&b, predicate, offset, shift, 0);
      EXPECT_FALSE(agx_nir_fuse_algebraic_late(b.shader));
      nir_validate_shader(b.shader, "unaligned store bounds");
      ralloc_free(b.shader);
   }
}

TEST(AlignedBounds, LeavesStandaloneAndUnrelatedPredicatesUnchanged)
{
   for (unsigned shift : {1u, 2u}) {
      for (unsigned control = 0; control < 4; ++control) {
         SCOPED_TRACE(shift);
         SCOPED_TRACE(control);
         uint32_t size = 1u << shift;
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "unrelated bounds predicate");
         nir_def *offset = agx_test_load_preamble(&b, 1, 32, 0);
         nir_def *other = agx_test_load_preamble(&b, 1, 32, 8);
         nir_def *bound = agx_test_load_preamble(&b, 1, 32, 4);
         nir_def *last = nir_iadd_imm(
            &b, nir_iand_imm(&b, offset, -size), size - 1);
         nir_def *predicate = nir_ult(&b, last, bound);
         nir_b2i32(&b, predicate);
         if (control == 1)
            indexed_guard_store(&b, predicate, other, shift, 0);
         if (control == 2)
            indexed_guard_store(&b, predicate, offset, shift, size);
         if (control == 3)
            indexed_guard_store(&b, predicate, offset, shift == 1 ? 2 : 1, 0);
         EXPECT_FALSE(agx_nir_fuse_algebraic_late(b.shader));
         nir_validate_shader(b.shader, "unrelated bounds predicate");
         ralloc_free(b.shader);
      }
   }
}
