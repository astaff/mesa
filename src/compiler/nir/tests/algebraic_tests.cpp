/*
 * Copyright © 2018 Intel Corporation
 * Copyright © 2021 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nir_test.h"
#include "nir_constant_expressions.h"
extern "C" {
#include "nir_search.h"
}
#include <vector>

namespace {

class algebraic_test_base : public nir_test {
protected:
   algebraic_test_base();

   virtual void run_pass()=0;

   void test_op(nir_op op, nir_def *src0, nir_def *src1, nir_def *src2,
                nir_def *src3, const char *desc);

   void test_2src_op(nir_op op, int64_t src0, int64_t src1);

   void require_one_alu(nir_op op);

   nir_variable *res_var;
};

algebraic_test_base::algebraic_test_base()
   : nir_test::nir_test("nir_opt_algebraic_test")
{
   res_var = nir_local_variable_create(b->impl, glsl_int_type(), "res");
}

void algebraic_test_base::test_op(nir_op op, nir_def *src0, nir_def *src1,
                                     nir_def *src2, nir_def *src3, const char *desc)
{
   nir_def *res_deref = &nir_build_deref_var(b, res_var)->def;

   /* create optimized expression */
   nir_intrinsic_instr *optimized_instr = nir_build_store_deref(
      b, res_deref, nir_build_alu(b, op, src0, src1, src2, src3), 0x1);

   run_pass();
   b->cursor = nir_after_cf_list(&b->impl->body);

   /* create reference expression */
   nir_intrinsic_instr *ref_instr = nir_build_store_deref(
      b, res_deref, nir_build_alu(b, op, src0, src1, src2, src3), 0x1);

   /* test equality */
   nir_opt_constant_folding(b->shader);

   ASSERT_TRUE(nir_src_is_const(ref_instr->src[1]));
   ASSERT_TRUE(nir_src_is_const(optimized_instr->src[1]));

   int32_t ref = nir_src_as_int(ref_instr->src[1]);
   int32_t optimized = nir_src_as_int(optimized_instr->src[1]);

   EXPECT_EQ(ref, optimized) << "Test input: " << desc;

   /* reset shader */
   exec_list_make_empty(&nir_start_block(b->impl)->instr_list);
   b->cursor = nir_after_cf_list(&b->impl->body);
}

void algebraic_test_base::test_2src_op(nir_op op, int64_t src0, int64_t src1)
{
   char desc[128];
   snprintf(desc, sizeof(desc), "%s(%" PRId64 ", %" PRId64 ")", nir_op_infos[op].name, src0, src1);
   test_op(op, nir_imm_int(b, src0), nir_imm_int(b, src1), NULL, NULL, desc);
}

void algebraic_test_base::require_one_alu(nir_op op)
{
   unsigned count = 0;
   nir_foreach_instr(instr, nir_start_block(b->impl)) {
      if (instr->type == nir_instr_type_alu) {
         ASSERT_TRUE(nir_instr_as_alu(instr)->op == op);
         ASSERT_EQ(count, 0);
         count++;
      }
   }
}

class nir_opt_algebraic_test : public algebraic_test_base {
protected:
   virtual void run_pass() {
      nir_opt_algebraic(b->shader);
   }
};

class nir_opt_idiv_const_test : public algebraic_test_base {
protected:
   virtual void run_pass() {
      nir_opt_idiv_const(b->shader, 8);
   }
};

class nir_opt_mqsad_test : public algebraic_test_base {
protected:
   virtual void run_pass() {
      nir_opt_mqsad(b->shader);
   }
};

static std::vector<unsigned> unary_match_condition_calls;

static bool
record_unary_match_condition(const nir_alu_instr *instr)
{
   unary_match_condition_calls.push_back(instr->def.index);
   return true;
}

TEST_F(nir_opt_algebraic_test, matched_variable_preserves_def_bits_and_swizzles)
{
   for (unsigned bits : {8u, 16u, 32u, 64u}) {
      for (unsigned components : {1u, 2u, 4u, 8u, 16u}) {
         for (unsigned depth : {2u, 4u, 8u}) {
            SCOPED_TRACE(bits);
            SCOPED_TRACE(components);
            SCOPED_TRACE(depth);
            b->cursor = nir_after_cf_list(&b->impl->body);
            nir_load_const_instr *input =
               nir_load_const_instr_create(b->shader, components, bits);
            for (unsigned c = 0; c < components; ++c)
               input->value[c] = nir_const_value_for_uint(17 + c, bits);
            nir_builder_instr_insert(b, &input->instr);
            nir_def *value = &input->def;
            std::vector<unsigned> indices;
            std::vector<std::vector<unsigned>> swizzles;
            for (unsigned level = 0; level < depth; ++level) {
               value = nir_inot(b, value);
               nir_alu_instr *alu = nir_def_as_alu(value);
               indices.push_back(value->index);
               swizzles.emplace_back();
               for (unsigned c = 0; c < components; ++c) {
                  unsigned swizzle =
                     (components - 1 - c + level) % components;
                  alu->src[0].swizzle[c] = swizzle;
                  swizzles.back().push_back(swizzle);
               }
            }
            nir_intrinsic_instr *store =
               nir_intrinsic_instr_create(b->shader, nir_intrinsic_store_ssbo);
            store->num_components = components;
            store->src[0] = nir_src_for_ssa(value);
            store->src[1] = nir_src_for_ssa(nir_imm_int(b, 0));
            store->src[2] = nir_src_for_ssa(nir_imm_int(b, 0));
            nir_intrinsic_set_write_mask(store, BITFIELD_MASK(components));
            nir_intrinsic_set_align_mul(store, 1);
            nir_builder_instr_insert(b, &store->instr);

            const unsigned variable = NIR_SEARCH_MAX_VARIABLES - 1;
            std::vector<nir_search_value_union> values(depth + 3);
            values[0].variable = {};
            values[0].variable.value.type = nir_search_value_variable;
            values[0].variable.variable = variable;
            values[0].variable.cond_index = -1;
            for (unsigned c = 0; c < NIR_MAX_VEC_COMPONENTS; ++c)
               values[0].variable.swizzle[c] = c;
            for (unsigned level = 1; level <= depth; ++level) {
               nir_search_expression &expr = values[level].expression;
               expr = {};
               expr.value.type = nir_search_value_expression;
               expr.opcode = nir_op_inot;
               expr.swizzle = -1;
               expr.comm_expr_idx = -1;
               expr.cond_index = 0;
               expr.srcs[0] = level - 1;
            }
            nir_search_constant &zero = values[depth + 1].constant;
            zero = {};
            zero.data.i = 0;
            zero.value.type = nir_search_value_constant;
            zero.value.bit_size = -int(variable) - 1;
            zero.type = nir_type_int;
            nir_search_expression &replacement = values[depth + 2].expression;
            replacement = {};
            replacement.value.type = nir_search_value_expression;
            replacement.value.bit_size = -int(variable) - 1;
            replacement.opcode = nir_op_iadd;
            replacement.swizzle = -1;
            replacement.comm_expr_idx = -1;
            replacement.cond_index = -1;
            replacement.srcs[0] = 0;
            replacement.srcs[1] = depth + 1;
            const uint16_t transition[] = {2};
            std::vector<per_op_table> operations(nir_num_search_ops);
            for (per_op_table &op : operations) {
               op.num_filtered_states = 1;
               op.table = transition;
            }
            const struct transform transforms[] = {
               {static_cast<uint16_t>(depth), static_cast<uint16_t>(depth + 2), 0}, {0, 0, ~0u},
            };
            const uint16_t offsets[] = {1, 1, 0};
            const nir_search_expression_cond conditions[] = {
               record_unary_match_condition,
            };
            nir_algebraic_table table = {};
            table.transforms = transforms;
            table.transform_offsets = offsets;
            table.pass_op_table = operations.data();
            table.values = values.data();
            table.expression_cond = conditions;
            const bool enabled[] = {true};
            unary_match_condition_calls.clear();
            ASSERT_TRUE(nir_algebraic_impl(b->impl, enabled, &table));
            ASSERT_GE(unary_match_condition_calls.size(), depth);
            for (unsigned level = 0; level < depth; ++level)
               EXPECT_EQ(unary_match_condition_calls[level],
                         indices[depth - 1 - level]);
            ASSERT_EQ(nir_src_bit_size(store->src[0]), bits);
            ASSERT_EQ(nir_src_num_components(store->src[0]), components);
            nir_validate_shader(b->shader, "matched variable representation");
            nir_opt_constant_folding(b->shader);
            ASSERT_TRUE(nir_src_is_const(store->src[0]));
            for (unsigned c = 0; c < components; ++c) {
               unsigned input_component = c;
               for (unsigned level = depth; level-- > 0;)
                  input_component = swizzles[level][input_component];
               EXPECT_EQ(nir_src_comp_as_uint(store->src[0], c),
                         17u + input_component);
            }
         }
      }
   }
}

static std::vector<const nir_alu_instr *> initial_worklist_visits;

static bool
record_initial_worklist_visit(const nir_alu_instr *instr)
{
   initial_worklist_visits.push_back(instr);
   return false;
}

TEST_F(nir_opt_algebraic_test, initial_worklist_order_and_flags)
{
   for (unsigned count : {0u, 1u, 7u, 8u, 9u, 15u, 16u, 17u,
                          31u, 32u, 33u, 127u, 128u, 129u}) {
      for (unsigned layout = 0; layout < 3; ++layout) {
         SCOPED_TRACE(count);
         SCOPED_TRACE(layout);
         nir_shader_compiler_options options = {};
         nir_builder current = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &options, "initial worklist order");
         ralloc_steal(b->shader, current.shader);
         nir_builder *build = &current;
         auto append_alu = [&](unsigned amount) {
            for (unsigned i = 0; i < amount; ++i)
               nir_inot(build, nir_imm_int(build, i + 3));
         };
         nir_load_local_invocation_index(build);
         if (layout == 0) {
            append_alu(count);
         } else if (layout == 1) {
            append_alu(count / 3);
            nir_if *branch = nir_push_if(build, nir_imm_true(build));
            append_alu(count / 3);
            nir_push_else(build, branch);
            append_alu(count - 2 * (count / 3));
            nir_pop_if(build, branch);
         } else {
            nir_loop *loop = nir_push_loop(build);
            nir_loop_add_continue_construct(loop);
            append_alu(count / 4);
            nir_if *branch = nir_push_if(build, nir_imm_true(build));
            append_alu(count / 4);
            nir_jump(build, nir_jump_break);
            nir_push_else(build, branch);
            append_alu(count / 4);
            nir_jump(build, nir_jump_continue);
            nir_pop_if(build, branch);
            nir_push_continue(build, loop);
            append_alu(count - 3 * (count / 4));
            nir_pop_loop(build, loop);
         }
         nir_validate_shader(current.shader, "before initial worklist");
         std::vector<const nir_alu_instr *> expected;
         nir_foreach_block_reverse(block, build->impl) {
            nir_foreach_instr_reverse(instr, block) {
               if (instr->type == nir_instr_type_alu) {
                  ASSERT_EQ(nir_instr_as_alu(instr)->op, nir_op_inot);
                  expected.push_back(nir_instr_as_alu(instr));
               }
            }
         }
         ASSERT_EQ(expected.size(), count);
         std::vector<nir_instr *> all;
         nir_foreach_block(block, build->impl) {
            nir_foreach_instr(instr, block) {
               instr->pass_flags = 0x5a;
               all.push_back(instr);
            }
         }
         const uint16_t transition[] = {2};
         std::vector<per_op_table> operations(nir_num_search_ops);
         for (per_op_table &op : operations) {
            op.num_filtered_states = 1;
            op.table = transition;
         }
         nir_search_value_union values[2] = {};
         values[0].variable = {};
         values[0].variable.value.type = nir_search_value_variable;
         values[0].variable.cond_index = -1;
         values[1].expression = {};
         values[1].expression.value.type = nir_search_value_expression;
         values[1].expression.opcode = nir_op_inot;
         values[1].expression.swizzle = -1;
         values[1].expression.comm_expr_idx = -1;
         values[1].expression.cond_index = 0;
         const struct transform transforms[] = {{1, 0, 0}, {0, 0, ~0u}};
         const uint16_t offsets[] = {1, 1, 0};
         const nir_search_expression_cond conditions[] = {
            record_initial_worklist_visit,
         };
         nir_algebraic_table table = {};
         table.transforms = transforms;
         table.transform_offsets = offsets;
         table.pass_op_table = operations.data();
         table.values = values;
         table.expression_cond = conditions;
         const bool enabled[] = {true};
         initial_worklist_visits.clear();
         EXPECT_FALSE(nir_algebraic_impl(build->impl, enabled, &table));
         EXPECT_EQ(initial_worklist_visits, expected);
         for (nir_instr *instr : all)
            EXPECT_EQ(instr->pass_flags, 0u);
         nir_validate_shader(current.shader, "after initial worklist");
         ralloc_free(current.shader);
      }
   }
}

TEST_F(nir_opt_algebraic_test, bounds_agx_unsigned_end_and_bound)
{
   for (unsigned bits : {8u, 16u, 32u, 64u}) {
      uint64_t mask = UINT64_MAX >> (64 - bits);
      uint64_t sign = UINT64_C(1) << (bits - 1);
      const uint64_t edge[] = {0, 1, 63, 64, 65, sign - 1, sign,
                              sign + 1, mask - 1, mask};
      const uint64_t data[] = {0, sign, mask, UINT64_C(0x7fc12345) & mask};
      for (uint64_t value : data) {
         for (uint64_t end : edge) {
            for (uint64_t bound : edge) {
               nir_const_value v[] = {
                  nir_const_value_for_uint(value, bits),
                  nir_const_value_for_uint(end, bits),
                  nir_const_value_for_uint(bound, bits),
               };
               nir_const_value *src[] = {&v[0], &v[1], &v[2]};
               nir_const_value out = {};
               uint16_t poison = 0;
               nir_eval_const_opcode(nir_op_bounds_agx, &out, &poison,
                                     1, bits, src, 0);
               EXPECT_EQ(poison, 0u);
               EXPECT_EQ(nir_const_value_as_uint(out, bits),
                         end <= bound ? value : 0)
                  << "bits=" << bits << " end=" << end << " bound=" << bound;
            }
         }
      }
   }
}

TEST_F(nir_opt_algebraic_test, bounds_agx_constant_fold_matches_unsigned_backend)
{
   nir_def *checked = nir_bounds_agx(b, nir_imm_int(b, 0xdeadbeef),
                                      nir_imm_int(b, UINT32_MAX),
                                      nir_imm_int(b, 64));
   nir_deref_instr *deref = nir_build_deref_var(b, res_var);
   nir_intrinsic_instr *store =
      nir_build_store_deref(b, &deref->def, checked, 1, (gl_access_qualifier)0);
   ASSERT_TRUE(nir_opt_constant_folding(b->shader));
   ASSERT_TRUE(nir_src_is_const(store->src[1]));
   EXPECT_EQ(nir_src_as_uint(store->src[1]), 0u);
}

TEST_F(nir_opt_algebraic_test, umod_pow2_src2)
{
   for (int i = 0; i <= 9; i++)
      test_2src_op(nir_op_umod, i, 4);
   test_2src_op(nir_op_umod, UINT32_MAX, 4);
}

TEST_F(nir_opt_algebraic_test, imod_pow2_src2)
{
   for (int i = -9; i <= 9; i++) {
      test_2src_op(nir_op_imod, i, 4);
      test_2src_op(nir_op_imod, i, -4);
      test_2src_op(nir_op_imod, i, INT32_MIN);
   }
   test_2src_op(nir_op_imod, INT32_MAX, 4);
   test_2src_op(nir_op_imod, INT32_MAX, -4);
   test_2src_op(nir_op_imod, INT32_MIN, 4);
   test_2src_op(nir_op_imod, INT32_MIN, -4);
   test_2src_op(nir_op_imod, INT32_MIN, INT32_MIN);
}

TEST_F(nir_opt_algebraic_test, irem_pow2_src2)
{
   for (int i = -9; i <= 9; i++) {
      test_2src_op(nir_op_irem, i, 4);
      test_2src_op(nir_op_irem, i, -4);
   }
   test_2src_op(nir_op_irem, INT32_MAX, 4);
   test_2src_op(nir_op_irem, INT32_MAX, -4);
   test_2src_op(nir_op_irem, INT32_MIN, 4);
   test_2src_op(nir_op_irem, INT32_MIN, -4);
}

TEST_F(nir_opt_algebraic_test, msad)
{
   options.lower_bitfield_extract = true;
   options.has_bfe = true;
   options.has_msad = true;

   nir_def *src0 = nir_load_var(b, nir_local_variable_create(b->impl, glsl_int_type(), "src0"));
   nir_def *src1 = nir_load_var(b, nir_local_variable_create(b->impl, glsl_int_type(), "src1"));

   /* This mimics the sequence created by vkd3d-proton. */
   nir_def *res = NULL;
   for (unsigned i = 0; i < 4; i++) {
      nir_def *ref = nir_ubitfield_extract(b, src0, nir_imm_int(b, i * 8), nir_imm_int(b, 8));
      nir_def *src = nir_ubitfield_extract(b, src1, nir_imm_int(b, i * 8), nir_imm_int(b, 8));
      nir_def *is_ref_zero = nir_ieq_imm(b, ref, 0);
      nir_def *abs_diff = nir_iabs(b, nir_isub(b, ref, src));
      nir_def *masked_diff = nir_bcsel(b, is_ref_zero, nir_imm_int(b, 0), abs_diff);
      if (res)
         res = nir_iadd(b, res, masked_diff);
      else
         res = masked_diff;
   }

   nir_store_var(b, res_var, res, 0x1);

   while (nir_opt_algebraic(b->shader)) {
      nir_opt_constant_folding(b->shader);
      nir_opt_dce(b->shader);
   }

   require_one_alu(nir_op_msad_4x8);
}

TEST_F(nir_opt_algebraic_test, commutative_match_does_not_leak_fp_math_ctrl)
{
   nir_def *src = nir_load_var(
      b, nir_local_variable_create(b->impl, glsl_float_type(), "src"));
   nir_def *inner = nir_fneg(b, src);
   nir_def_as_alu(inner)->fp_math_ctrl = nir_fp_exact;
   nir_def *outer = nir_fneg(b, inner);
   nir_def *result = nir_fmin(b, outer, inner);
   nir_variable *result_var =
      nir_local_variable_create(b->impl, glsl_float_type(), "result");
   nir_intrinsic_instr *store = nir_build_store_deref(
      b, &nir_build_deref_var(b, result_var)->def, result, 0x1);

   ASSERT_TRUE(nir_opt_algebraic(b->shader));
   nir_alu_instr *optimized = nir_src_as_alu(store->src[1]);
   ASSERT_NE(optimized, nullptr);
   EXPECT_EQ(optimized->op, nir_op_fneg);
   EXPECT_EQ(optimized->fp_math_ctrl & nir_fp_exact, 0);
}

TEST_F(nir_opt_mqsad_test, mqsad)
{
   options.lower_bitfield_extract = true;
   options.has_bfe = true;
   options.has_msad = true;
   options.has_shfr32 = true;

   nir_def *ref = nir_load_var(b, nir_local_variable_create(b->impl, glsl_int_type(), "ref"));
   nir_def *src = nir_load_var(b, nir_local_variable_create(b->impl, glsl_ivec_type(2), "src"));
   nir_def *accum = nir_load_var(b, nir_local_variable_create(b->impl, glsl_ivec_type(4), "accum"));

   nir_def *srcx = nir_channel(b, src, 0);
   nir_def *srcy = nir_channel(b, src, 1);

   nir_def *res[4];
   for (unsigned i = 0; i < 4; i++) {
      nir_def *src1 = srcx;
      switch (i) {
      case 0:
         break;
      case 1:
         src1 = nir_bitfield_select(b, nir_imm_int(b, 0xff000000), nir_ishl_imm(b, srcy, 24),
                                    nir_ushr_imm(b, srcx, 8));
         break;
      case 2:
         src1 = nir_bitfield_select(b, nir_imm_int(b, 0xffff0000), nir_ishl_imm(b, srcy, 16),
                                    nir_extract_u16(b, srcx, nir_imm_int(b, 1)));
         break;
      case 3:
         src1 = nir_bitfield_select(b, nir_imm_int(b, 0xffffff00), nir_ishl_imm(b, srcy, 8),
                                    nir_extract_u8_imm(b, srcx, 3));
         break;
      }

      res[i] = nir_msad_4x8(b, ref, src1, nir_channel(b, accum, i));
   }

   nir_store_var(b, nir_local_variable_create(b->impl, glsl_ivec_type(4), "res"), nir_vec(b, res, 4), 0xf);

   while (nir_opt_algebraic(b->shader)) {
      nir_opt_constant_folding(b->shader);
      nir_opt_dce(b->shader);
   }

   ASSERT_TRUE(nir_opt_mqsad(b->shader));
   nir_opt_copy_prop(b->shader);
   nir_opt_dce(b->shader);

   require_one_alu(nir_op_mqsad_4x8);
}

TEST_F(nir_opt_idiv_const_test, umod)
{
   for (uint32_t d : {16u, 17u, 0u, UINT32_MAX}) {
      for (int i = 0; i <= 40; i++)
         test_2src_op(nir_op_umod, i, d);
      for (int i = 0; i < 20; i++)
         test_2src_op(nir_op_umod, UINT32_MAX - i, d);
   }
}

TEST_F(nir_opt_idiv_const_test, imod)
{
   for (int32_t d : {16, -16, 17, -17, 0, INT32_MIN, INT32_MAX}) {
      for (int i = -40; i <= 40; i++)
         test_2src_op(nir_op_imod, i, d);
      for (int i = 0; i < 20; i++)
         test_2src_op(nir_op_imod, INT32_MIN + i, d);
      for (int i = 0; i < 20; i++)
         test_2src_op(nir_op_imod, INT32_MAX - i, d);
   }
}

TEST_F(nir_opt_idiv_const_test, irem)
{
   for (int32_t d : {16, -16, 17, -17, 0, INT32_MIN, INT32_MAX}) {
      for (int i = -40; i <= 40; i++)
         test_2src_op(nir_op_irem, i, d);
      for (int i = 0; i < 20; i++)
         test_2src_op(nir_op_irem, INT32_MIN + i, d);
      for (int i = 0; i < 20; i++)
         test_2src_op(nir_op_irem, INT32_MAX - i, d);
   }
}

}
