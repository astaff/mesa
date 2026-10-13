/* SPDX-License-Identifier: MIT */
#include <array>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
}

static nir_shader *
make_program(nir_op op, uint32_t a_mask, uint32_t b_mask, unsigned form = 0)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "bounded integer arithmetic");
   nir_def *a = agx_test_load_preamble(&b, 1, 32, 0);
   nir_def *c = agx_test_load_preamble(&b, 1, 32, 2);
   a = nir_iand_imm(&b, a, a_mask);
   c = nir_iand_imm(&b, c, b_mask);
   nir_def *wide_a;
   if (form == 1)
      wide_a = nir_pack_64_2x32_split(&b, a, nir_imm_int(&b, 0));
   else if (form == 2)
      wide_a = nir_i2i64(&b, a);
   else if (form == 3)
      wide_a = agx_test_load_preamble(&b, 1, 64, 4);
   else
      wide_a = nir_u2u64(&b, a);
   bool shift = op == nir_op_ushr || op == nir_op_ishr || op == nir_op_ishl;
   nir_def *wide_c = shift ? c : nir_u2u64(&b, c);
   if (form == 4)
      wide_c = nir_b2i64(&b, nir_ieq_imm(&b, c, 0));
   nir_def *result = nir_build_alu(&b, op, wide_a, wide_c, NULL, NULL);
   nir_store_global(&b, result, nir_imm_int64(&b, 0), .align_mul = 8);
   nir_validate_shader(b.shader, "before integer narrowing");
   return b.shader;
}

static uint64_t
evaluate(nir_shader *shader, uint32_t a, uint32_t b, uint64_t wide)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_builder builder = nir_builder_create(impl);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_preamble)
            continue;
         unsigned base = nir_intrinsic_base(intr);
         uint64_t bits = base == 0 ? a : base == 2 ? b : wide;
         builder.cursor = nir_before_instr(instr);
         nir_def_rewrite_uses(
            &intr->def, nir_imm_intN_t(&builder, bits, intr->def.bit_size));
         nir_instr_remove(instr);
      }
   }
   bool progress;
   do {
      progress = nir_opt_constant_folding(shader);
      progress |= nir_opt_copy_prop(shader);
   } while (progress);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_global) {
            EXPECT_TRUE(nir_src_is_const(intr->src[0]));
            if (nir_src_is_const(intr->src[0]))
               return nir_src_as_uint(intr->src[0]);
         }
      }
   }
   ADD_FAILURE();
   return 0;
}

static uint64_t
reference(nir_op op, uint64_t a, uint64_t b)
{
   switch (op) {
   case nir_op_iadd:
      return a + b;
   case nir_op_imul:
      return a * b;
   case nir_op_iand:
      return a & b;
   case nir_op_ior:
      return a | b;
   case nir_op_ixor:
      return a ^ b;
   case nir_op_ushr:
      return a >> (b & 63);
   case nir_op_ishr:
      return uint64_t(int64_t(a) >> (b & 63));
   case nir_op_ishl:
      return a << (b & 63);
   default:
      ADD_FAILURE();
      return 0;
   }
}

static void
check(nir_op op, uint32_t a_mask, uint32_t b_mask, unsigned form,
      bool should_narrow)
{
   SCOPED_TRACE(op);
   SCOPED_TRACE(form);
   nir_shader *shader = make_program(op, a_mask, b_mask, form);
   EXPECT_EQ(agx_nir_scalarize_int64(shader), should_narrow);
   nir_validate_shader(shader, "after integer narrowing");
   uint32_t state = 13;
   auto random = [&state]() {
      state = state * 1664525u + 1013904223u;
      return state;
   };
   for (unsigned i = 0; i < 384; ++i) {
      uint32_t a =
         i < 6 ? std::array<uint32_t, 6>{0,          1,          0x7fffffff,
                                         0x80000000, 0xfffffffe, 0xffffffff}[i]
               : random();
      uint32_t b = i < 128 ? i : random();
      uint64_t wide = uint64_t(random()) << 32 | random();
      uint64_t x = a & a_mask;
      uint64_t y = b & b_mask;
      if (form == 2)
         x = uint64_t(int64_t(int32_t(x)));
      if (form == 3)
         x = wide;
      if (form == 4)
         y = y == 0;
      nir_shader *copy = nir_shader_clone(NULL, shader);
      EXPECT_EQ(evaluate(copy, a, b, wide), reference(op, x, y));
      ralloc_free(copy);
   }
   ralloc_free(shader);
}

TEST(ScalarizeInt64, PreservesEveryShiftBoundary)
{
   for (nir_op op : {nir_op_ushr, nir_op_ishr}) {
      for (unsigned form : {0u, 1u}) {
         check(op, UINT32_MAX, UINT32_MAX, form, true);
         check(op, UINT32_MAX, 31, form, true);
      }
      check(op, UINT32_MAX, UINT32_MAX, 2, false);
      check(op, 0x7fffffff, UINT32_MAX, 2, true);
   }
   check(nir_op_ishl, 0xffff, 15, 0, true);
   check(nir_op_ishl, UINT32_MAX, 31, 0, false);
   check(nir_op_ishl, 1, 63, 0, false);
}

TEST(ScalarizeInt64, PreservesCarryAndOverflow)
{
   check(nir_op_iadd, 0xffffff, 1, 0, false);
   check(nir_op_iadd, UINT32_MAX, UINT32_MAX, 0, false);
   check(nir_op_iadd, 0xffffff, UINT32_MAX, 4, false);
   check(nir_op_iadd, 0x7fffffff, 1, 2, false);
   check(nir_op_iadd, UINT32_MAX, 1, 2, false);
   check(nir_op_imul, 0xffff, 0xffff, 0, false);
   check(nir_op_imul, 0xffffff, 0xffffff, 0, false);
}

TEST(ScalarizeInt64, PreservesFullWidthMaskedInputs)
{
   for (nir_op op : {nir_op_iand, nir_op_ior, nir_op_ixor})
      check(op, UINT32_MAX, UINT32_MAX, 0, true);
   check(nir_op_iand, UINT32_MAX, 0xffff, 3, true);
   check(nir_op_ior, UINT32_MAX, 0xffff, 3, false);
   check(nir_op_ixor, UINT32_MAX, 0xffff, 3, false);
}

TEST(ScalarizeInt64, PreservesConstantVectorsAndFullWidthValues)
{
   const uint64_t constants[] = {0xffffff, 0xffffffff, 0x100000000,
                                 UINT64_C(0x8000000000000000)};
   for (nir_op op : {nir_op_ushr, nir_op_ishr}) {
      for (unsigned component = 0; component < 4; ++component) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "wide constant shift");
         nir_const_value raw[4] = {};
         for (unsigned i = 0; i < 4; ++i)
            raw[i].u64 = constants[i];
         nir_def *value =
            nir_channel(&b, nir_build_imm(&b, 4, 64, raw), component);
         nir_def *count = agx_test_load_preamble(&b, 1, 32, 2);
         nir_def *result = nir_build_alu(&b, op, value, count, NULL, NULL);
         nir_store_global(&b, result, nir_imm_int64(&b, 0), .align_mul = 8);
         EXPECT_EQ(agx_nir_scalarize_int64(b.shader), component < 2);
         nir_validate_shader(b.shader, "wide constant narrowing");
         for (uint32_t n = 0; n < 128; ++n) {
            nir_shader *copy = nir_shader_clone(NULL, b.shader);
            EXPECT_EQ(evaluate(copy, 0, n, 0),
                      reference(op, constants[component], n));
            ralloc_free(copy);
         }
         ralloc_free(b.shader);
      }
   }
}
