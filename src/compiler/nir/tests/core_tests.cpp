/*
 * Copyright © 2018 Intel Corporation
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
extern "C" {
#include "nir_instr_set.h"
#include "nir_worklist.h"
#include "nir_control_flow.h"
}
#include <vector>
#include <deque>
#include <algorithm>
#include <string>

namespace {

class nir_core_test : public nir_test {
protected:
   nir_core_test()
      : nir_test::nir_test("nir_core_test")
   {
   }

   bool shader_contains_def(nir_def *def);
};

struct contains_def_state {
   nir_def *def;
   bool found;
};

static bool
contains_def_cb(nir_def *def, void *_state)
{
   struct contains_def_state *state = (struct contains_def_state *)_state;
   if (def == state->def)
      state->found = true;

   return true;
}

bool
nir_core_test::shader_contains_def(nir_def *def)
{
   nir_foreach_block(block, b->impl) {
      nir_foreach_instr(instr, block) {
         struct contains_def_state state = {
            def, false
         };
         nir_foreach_def(instr, contains_def_cb, &state);
         if (state.found)
            return true;
      }
   }
   return false;
}

TEST_F(nir_core_test, nir_instr_free_and_dce_test)
{
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *one = nir_imm_int(b, 1);
   nir_def *add01 = nir_iadd(b, zero, one);
   nir_def *add11 = nir_iadd(b, one, one);

   nir_cursor c = nir_instr_free_and_dce(nir_def_instr(add01));
   ASSERT_FALSE(shader_contains_def(add01));
   ASSERT_TRUE(shader_contains_def(add11));
   ASSERT_FALSE(shader_contains_def(zero));
   ASSERT_TRUE(shader_contains_def(one));

   ASSERT_TRUE(nir_cursors_equal(c, nir_before_def(add11)));

   nir_validate_shader(b->shader, "after remove_and_dce");
}

TEST_F(nir_core_test, nir_instr_free_and_dce_all_test)
{
   nir_def *one = nir_imm_int(b, 1);
   nir_def *add = nir_iadd(b, one, one);

   nir_cursor c = nir_instr_free_and_dce(nir_def_instr(add));
   ASSERT_FALSE(shader_contains_def(add));
   ASSERT_FALSE(shader_contains_def(one));

   ASSERT_TRUE(nir_cursors_equal(c, nir_before_impl(b->impl)));

   nir_validate_shader(b->shader, "after remove_and_dce");
}

TEST_F(nir_core_test, nir_instr_free_and_dce_multiple_src_test)
{
   nir_def *one = nir_imm_int(b, 1);
   nir_def *add = nir_iadd(b, one, one);

   /* This risks triggering removing add multiple times, which can segfault in
    * nir_instr_remove for instructions with srcs. */
   nir_def *add2 = nir_iadd(b, add, add);

   nir_cursor c = nir_instr_free_and_dce(nir_def_instr(add2));
   ASSERT_FALSE(shader_contains_def(add2));
   ASSERT_FALSE(shader_contains_def(add));
   ASSERT_FALSE(shader_contains_def(one));

   ASSERT_TRUE(nir_cursors_equal(c, nir_before_impl(b->impl)));

   nir_validate_shader(b->shader, "after remove_and_dce");
}

TEST_F(nir_core_test, instruction_worklist_preserves_fifo_across_growth_and_wrap)
{
   std::vector<nir_instr *> values;
   for (unsigned i = 0; i < 17; ++i)
      values.push_back(nir_def_instr(nir_imm_int(b, i)));

   for (bool near_overflow : {false, true}) {
      SCOPED_TRACE(near_overflow);
      nir_instr_worklist worklist;
      ASSERT_TRUE(nir_instr_worklist_init(&worklist));
      const uint32_t start = near_overflow ? 0u - worklist.instr_vec.size : 0u;
      worklist.instr_vec.head = worklist.instr_vec.tail = start;
      std::deque<nir_instr *> expected;
      unsigned next_value = 0;
      auto push = [&]() {
         nir_instr *value = values[next_value++ % values.size()];
         nir_instr_worklist_push_tail(&worklist, value);
         expected.push_back(value);
         EXPECT_EQ(nir_instr_worklist_length(&worklist), expected.size());
      };
      auto pop = [&]() {
         ASSERT_FALSE(expected.empty());
         EXPECT_EQ(nir_instr_worklist_pop_head(&worklist), expected.front());
         expected.pop_front();
         EXPECT_EQ(nir_instr_worklist_length(&worklist), expected.size());
      };
      EXPECT_EQ(nir_instr_worklist_pop_head(&worklist), nullptr);
      for (unsigned i = 0; i < 9; ++i)
         push();
      for (unsigned i = 0; i < 9; ++i)
         pop();
      EXPECT_EQ(nir_instr_worklist_pop_head(&worklist), nullptr);
      for (unsigned phase = 0; phase < 4; ++phase) {
         for (unsigned i = 0; i < 17 + 13 * phase; ++i)
            push();
         for (unsigned i = 0; i < 9; ++i)
            pop();
         for (unsigned i = 0; i < 61 + 7 * phase; ++i)
            push();
         while (!expected.empty())
            pop();
         EXPECT_TRUE(nir_instr_worklist_is_empty(&worklist));
         EXPECT_EQ(nir_instr_worklist_pop_head(&worklist), nullptr);
      }
      nir_instr_worklist_fini(&worklist);
   }
   nir_validate_shader(b->shader, "worklist FIFO");
}

static nir_alu_instr *
source_hash_alu(nir_builder *build, nir_op op, unsigned components, unsigned bits,
                nir_def *a, nir_def *c)
{
   nir_alu_instr *alu = nir_alu_instr_create(build->shader, op);
   alu->src[0].src = nir_src_for_ssa(a);
   alu->src[1].src = nir_src_for_ssa(c);
   nir_def_init(&alu->instr, &alu->def, components, bits);
   nir_builder_instr_insert(build, &alu->instr);
   return alu;
}

TEST_F(nir_core_test, alu_hash_preserves_commutative_source_keys)
{
   for (unsigned bits : {8u, 16u, 32u, 64u}) {
      for (unsigned components : {1u, 2u, 3u, 4u, 8u, 16u}) {
         for (bool alias : {false, true}) {
            for (unsigned flags = 0; flags < 4; ++flags) {
               SCOPED_TRACE(bits);
               SCOPED_TRACE(components);
               SCOPED_TRACE(alias);
               SCOPED_TRACE(flags);
               nir_def *a = nir_undef(b, 16, bits);
               nir_def *c = alias ? a : nir_undef(b, 16, bits);
               nir_alu_instr *left =
                  source_hash_alu(b, nir_op_iadd, components, bits, a, c);
               nir_alu_instr *right =
                  source_hash_alu(b, nir_op_iadd, components, bits, c, a);
               left->no_signed_wrap = right->no_signed_wrap = flags & 1;
               left->no_unsigned_wrap = right->no_unsigned_wrap = (flags & 2) != 0;
               for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; ++i) {
                  left->src[0].swizzle[i] = i < components ? (i + 1) % 16 : 0;
                  left->src[1].swizzle[i] = i < components ? (i + 7) % 16 : 1;
                  right->src[0].swizzle[i] = i < components ? left->src[1].swizzle[i] : 3;
                  right->src[1].swizzle[i] = i < components ? left->src[0].swizzle[i] : 4;
               }
               ASSERT_TRUE(nir_instrs_equal(&left->instr, &right->instr));
               struct set instrs;
               nir_instr_set_init(&instrs, nullptr);
               EXPECT_EQ(instrs.key_hash_function(&left->instr),
                         instrs.key_hash_function(&right->instr));
               _mesa_set_search_or_add(&instrs, &left->instr, nullptr);
               bool found = false;
               set_entry *entry =
                  _mesa_set_search_or_add(&instrs, &right->instr, &found);
               EXPECT_TRUE(found);
               EXPECT_EQ(entry->key, &left->instr);
               nir_instr_set_fini(&instrs);
            }
         }
      }
   }
   nir_validate_shader(b->shader, "commutative source hash keys");
}

TEST_F(nir_core_test, alu_hash_preserves_noncommutative_order)
{
   nir_def *a = nir_undef(b, 16, 32);
   nir_def *c = nir_undef(b, 16, 32);
   struct set instrs;
   nir_instr_set_init(&instrs, nullptr);
   bool any_hash_differs = false;
   for (unsigned i = 0; i < 16; ++i) {
      nir_alu_instr *left = source_hash_alu(b, nir_op_ishl, 1, 32, a, c);
      nir_alu_instr *right = source_hash_alu(b, nir_op_ishl, 1, 32, c, a);
      left->src[0].swizzle[0] = right->src[1].swizzle[0] = i;
      left->src[1].swizzle[0] = right->src[0].swizzle[0] = (i + 1) % 16;
      EXPECT_FALSE(nir_instrs_equal(&left->instr, &right->instr));
      any_hash_differs |= instrs.key_hash_function(&left->instr) !=
                          instrs.key_hash_function(&right->instr);
   }
   EXPECT_TRUE(any_hash_differs);
   nir_instr_set_fini(&instrs);
   nir_validate_shader(b->shader, "ordered source hash keys");
}

TEST_F(nir_core_test, alu_hash_preserves_fixed_input_components)
{
   nir_def *a = nir_undef(b, 16, 32);
   nir_def *c = nir_undef(b, 16, 32);
   struct set instrs;
   nir_instr_set_init(&instrs, nullptr);
   std::vector<uint32_t> hashes;
   for (unsigned i = 0; i < 16; ++i) {
      nir_alu_instr *dot = source_hash_alu(b, nir_op_fdot4, 1, 32, a, c);
      for (unsigned channel = 0; channel < 4; ++channel) {
         dot->src[0].swizzle[channel] = channel;
         dot->src[1].swizzle[channel] = channel + 4;
      }
      dot->src[0].swizzle[3] = i;
      hashes.push_back(instrs.key_hash_function(&dot->instr));
   }
   bool any_hash_differs = false;
   for (uint32_t hash : hashes)
      any_hash_differs |= hash != hashes[0];
   EXPECT_TRUE(any_hash_differs);
   nir_instr_set_fini(&instrs);
   nir_validate_shader(b->shader, "fixed source component hash keys");
}

static nir_alu_instr *
clone_sparse_sum(nir_builder *build, unsigned holes, unsigned value,
                 bool keep_definitions = false)
{
   nir_def *a = nir_imm_int(build, value);
   for (unsigned i = 0; i < holes; ++i) {
      nir_def *unused = nir_undef(build, 1, 32);
      nir_instr_remove(nir_def_instr(unused));
      nir_instr_free(nir_def_instr(unused));
      build->cursor = nir_after_def(a);
   }
   if (keep_definitions) {
      for (unsigned i = 0; i < holes; ++i)
         nir_undef(build, 1, 32);
   }
   return nir_def_as_alu(nir_iadd(build, a, nir_imm_int(build, value + 7)));
}

static nir_function_impl *
clone_find_impl(nir_shader *shader, const char *name)
{
   nir_foreach_function(function, shader) {
      if (strcmp(function->name, name) == 0)
         return function->impl;
   }
   return nullptr;
}

static nir_alu_instr *
clone_find_sum(nir_function_impl *impl)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_alu &&
             nir_instr_as_alu(instr)->op == nir_op_iadd)
            return nir_instr_as_alu(instr);
      }
   }
   return nullptr;
}

TEST_F(nir_core_test, clone_sparse_definitions_and_multiple_functions)
{
   struct record {
      nir_function_impl *impl;
      nir_alu_instr *sum;
      unsigned capacity;
      unsigned value;
   };
   std::vector<record> records;
   for (unsigned holes : {129u, 0u, 33u, 1u, 1024u, 31u, 32u, 63u,
                          64u, 65u, 127u, 128u}) {
      for (bool keep : {false, true}) {
         std::string name = "sparse_clone_" + std::to_string(holes) +
                            (keep ? "_kept" : "_sparse");
         nir_function *function = nir_function_create(b->shader, name.c_str());
         nir_function_impl *impl = nir_function_impl_create(function);
         nir_builder build = nir_builder_create(impl);
         build.cursor = nir_after_block(nir_start_block(impl));
         unsigned value = 13 + records.size();
         nir_alu_instr *sum = clone_sparse_sum(&build, holes, value, keep);
         records.push_back({impl, sum, impl->ssa_alloc, value});
      }
   }
   nir_validate_shader(b->shader, "sparse clone source");
   nir_shader *copy = nir_shader_clone(nullptr, b->shader);
   nir_validate_shader(copy, "sparse clone result");
   nir_validate_ssa_dominance(copy, "sparse clone result");
   for (const record &record : records) {
      nir_function_impl *impl = clone_find_impl(copy, record.impl->function->name);
      ASSERT_NE(impl, nullptr);
      EXPECT_NE(impl, record.impl);
      nir_alu_instr *sum = clone_find_sum(impl);
      ASSERT_NE(sum, nullptr);
      EXPECT_NE(sum, record.sum);
      EXPECT_EQ(nir_src_as_uint(sum->src[0].src), record.value);
      EXPECT_EQ(nir_src_as_uint(sum->src[1].src), record.value + 7);
      for (unsigned i = 0; i < 2; ++i) {
         EXPECT_NE(sum->src[i].src.ssa, record.sum->src[i].src.ssa);
         EXPECT_EQ(nir_def_block(sum->src[i].src.ssa)->impl, impl);
      }
      EXPECT_EQ(record.impl->ssa_alloc, record.capacity);
      EXPECT_EQ(record.sum->src[0].src.ssa->index, 0u);
      EXPECT_EQ(record.sum->def.index, record.capacity - 1);
   }
   ralloc_free(copy);
}

TEST_F(nir_core_test, clone_loop_carried_phi_sources)
{
   nir_def *initial = nir_imm_int(b, 0);
   nir_def *step = nir_imm_int(b, 1);
   nir_block *preheader = nir_cursor_current_block(b->cursor);
   nir_loop *loop = nir_push_loop(b);
   nir_phi_instr *phi = nir_phi_instr_create(b->shader);
   nir_def_init(&phi->instr, &phi->def, 1, 32);
   nir_def *next = nir_iadd(b, &phi->def, step);
   nir_block *backedge = nir_def_block(next);
   nir_pop_loop(b, loop);
   nir_phi_instr_add_src(phi, preheader, initial);
   nir_phi_instr_add_src(phi, backedge, next);
   b->cursor = nir_before_block(backedge);
   nir_builder_instr_insert(b, &phi->instr);
   nir_validate_shader(b->shader, "loop clone source");
   nir_shader *copy = nir_shader_clone(nullptr, b->shader);
   nir_validate_shader(copy, "loop clone result");
   nir_validate_ssa_dominance(copy, "loop clone result");
   nir_function_impl *impl = nir_shader_get_entrypoint(copy);
   nir_alu_instr *sum = clone_find_sum(impl);
   ASSERT_NE(sum, nullptr);
   nir_phi_instr *new_phi = nir_src_as_phi(sum->src[0].src);
   ASSERT_NE(new_phi, nullptr);
   EXPECT_NE(new_phi, phi);
   EXPECT_EQ(nir_src_as_uint(sum->src[1].src), 1u);
   unsigned initial_edges = 0, back_edges = 0;
   nir_foreach_phi_src(src, new_phi) {
      if (src->src.ssa == &sum->def) {
         EXPECT_EQ(src->pred, sum->instr.block);
         ++back_edges;
      } else {
         EXPECT_EQ(nir_src_as_uint(src->src), 0u);
         EXPECT_NE(src->pred, preheader);
         EXPECT_EQ(src->pred->impl, impl);
         ++initial_edges;
      }
   }
   EXPECT_EQ(initial_edges, 1u);
   EXPECT_EQ(back_edges, 1u);
   ralloc_free(copy);
}

TEST_F(nir_core_test, clone_preserves_caller_owned_definition_remap)
{
   nir_alu_instr *old_sum = clone_sparse_sum(b, 65, 77, true);
   hash_table *remap = _mesa_pointer_hash_table_create(nullptr);
   unsigned marker = 0, value = 9;
   _mesa_hash_table_insert(remap, &marker, &value);
   nir_function *function = nir_function_create(b->shader, "remapped_clone");
   nir_function_impl *impl =
      nir_function_impl_clone_remap_globals(b->shader, b->impl, remap);
   nir_function_set_impl(function, impl);
   nir_alu_instr *sum = clone_find_sum(impl);
   ASSERT_NE(sum, nullptr);
   const nir_def *old_defs[] = {&old_sum->def, old_sum->src[0].src.ssa,
                                old_sum->src[1].src.ssa};
   const nir_def *new_defs[] = {&sum->def, sum->src[0].src.ssa,
                                sum->src[1].src.ssa};
   for (unsigned i = 0; i < 3; ++i) {
      hash_entry *entry = _mesa_hash_table_search(remap, old_defs[i]);
      ASSERT_NE(entry, nullptr);
      EXPECT_EQ(entry->data, new_defs[i]);
   }
   hash_entry *entry = _mesa_hash_table_search(remap, &marker);
   ASSERT_NE(entry, nullptr);
   EXPECT_EQ(entry->data, &value);
   nir_validate_shader(b->shader, "caller remap clone");
   nir_validate_ssa_dominance(b->shader, "caller remap clone");
   _mesa_hash_table_destroy(remap, nullptr);
}

TEST_F(nir_core_test, clone_instruction_sources_keep_remap_fallback)
{
   nir_alu_instr *old_sum = clone_sparse_sum(b, 1, 3);
   nir_def *replacement = nir_imm_int(b, 99);
   hash_table *remap = _mesa_pointer_hash_table_create(nullptr);
   _mesa_hash_table_insert(remap, old_sum->src[0].src.ssa, replacement);
   nir_instr *instr = nir_instr_clone_deep(b->shader, &old_sum->instr, remap);
   nir_builder_instr_insert(b, instr);
   nir_alu_instr *sum = nir_instr_as_alu(instr);
   EXPECT_EQ(sum->src[0].src.ssa, replacement);
   EXPECT_EQ(sum->src[1].src.ssa, old_sum->src[1].src.ssa);
   hash_entry *entry = _mesa_hash_table_search(remap, &old_sum->def);
   ASSERT_NE(entry, nullptr);
   EXPECT_EQ(entry->data, &sum->def);
   nir_alu_instr *plain = nir_alu_instr_clone(b->shader, old_sum);
   nir_builder_instr_insert(b, &plain->instr);
   EXPECT_EQ(plain->src[0].src.ssa, old_sum->src[0].src.ssa);
   EXPECT_EQ(plain->src[1].src.ssa, old_sum->src[1].src.ssa);
   nir_validate_shader(b->shader, "instruction clone fallback");
   nir_validate_ssa_dominance(b->shader, "instruction clone fallback");
   _mesa_hash_table_destroy(remap, nullptr);
}

TEST_F(nir_core_test, clone_cf_list_preserves_external_sources_and_remap)
{
   nir_alu_instr *old_sum = clone_sparse_sum(b, 33, 17);
   nir_cf_list source, copy;
   nir_cursor gap = nir_cf_extract(&source, nir_before_instr(&old_sum->instr),
                                   nir_after_instr(&old_sum->instr));
   hash_table *remap = _mesa_pointer_hash_table_create(nullptr);
   nir_cf_list_clone(&copy, &source, &b->impl->cf_node, remap);
   hash_entry *entry = _mesa_hash_table_search(remap, &old_sum->def);
   ASSERT_NE(entry, nullptr);
   nir_def *new_def = static_cast<nir_def *>(entry->data);
   nir_alu_instr *sum = nir_def_as_alu(new_def);
   EXPECT_NE(sum, old_sum);
   EXPECT_EQ(sum->src[0].src.ssa, old_sum->src[0].src.ssa);
   EXPECT_EQ(sum->src[1].src.ssa, old_sum->src[1].src.ssa);
   nir_cf_reinsert(&source, gap);
   nir_cf_reinsert(&copy, nir_after_impl(b->impl));
   nir_validate_shader(b->shader, "CF list clone fallback");
   nir_validate_ssa_dominance(b->shader, "CF list clone fallback");
   _mesa_hash_table_destroy(remap, nullptr);
}

TEST_F(nir_core_test, clone_local_declarations_preserve_mapping_context)
{
   struct local_case {
      nir_function_impl *impl;
      unsigned capacity;
      std::vector<nir_variable *> variables;
   };
   std::vector<local_case> cases;
   for (unsigned holes : {126u, 127u}) {
      for (unsigned count : {0u, 12u, 13u, 16u, 64u}) {
         std::string name = "mapped_locals_" + std::to_string(holes) + "_" +
                            std::to_string(count);
         nir_function *function = nir_function_create(b->shader, name.c_str());
         nir_function_impl *impl = nir_function_impl_create(function);
         nir_builder build = nir_builder_create(impl);
         build.cursor = nir_after_block(nir_start_block(impl));
         clone_sparse_sum(&build, holes, 47);
         std::vector<nir_variable *> variables;
         for (unsigned i = 0; i < count; ++i) {
            std::string variable_name = "local_" + std::to_string(i);
            variables.push_back(nir_local_variable_create(
               impl, glsl_int_type(), variable_name.c_str()));
         }
         cases.push_back({impl, impl->ssa_alloc, variables});
      }
   }
   nir_validate_shader(b->shader, "local-context clone source");
   nir_shader *copy = nir_shader_clone(nullptr, b->shader);
   nir_validate_shader(copy, "local-context clone result");
   nir_validate_ssa_dominance(copy, "local-context clone result");
   for (const local_case &source : cases) {
      nir_function_impl *impl = clone_find_impl(copy, source.impl->function->name);
      ASSERT_NE(impl, nullptr);
      std::vector<nir_variable *> variables;
      foreach_list_typed(nir_variable, var, node, &impl->locals)
         variables.push_back(var);
      ASSERT_EQ(variables.size(), source.variables.size());
      for (unsigned i = 0; i < variables.size(); ++i) {
         EXPECT_NE(variables[i], source.variables[i]);
         EXPECT_EQ(variables[i]->type, source.variables[i]->type);
         EXPECT_STREQ(variables[i]->name, source.variables[i]->name);
         EXPECT_EQ(static_cast<unsigned>(variables[i]->data.mode),
                   static_cast<unsigned>(nir_var_function_temp));
      }
      nir_alu_instr *sum = clone_find_sum(impl);
      ASSERT_NE(sum, nullptr);
      EXPECT_EQ(nir_src_as_uint(sum->src[0].src), 47u);
      EXPECT_EQ(nir_src_as_uint(sum->src[1].src), 54u);
      EXPECT_EQ(source.impl->ssa_alloc, source.capacity);
   }
   ralloc_free(copy);
}

TEST_F(nir_core_test, clone_body_blocks_preserve_edges_and_conditions)
{
   struct block_case {
      nir_function_impl *impl;
      unsigned capacity;
      unsigned if_count;
   };
   std::vector<block_case> cases;
   for (unsigned holes : {125u, 126u}) {
      for (unsigned count : {0u, 3u, 4u, 8u, 9u, 10u, 32u}) {
         std::string name = "mapped_blocks_" + std::to_string(holes) + "_" +
                            std::to_string(count);
         nir_function *function = nir_function_create(b->shader, name.c_str());
         nir_function_impl *impl = nir_function_impl_create(function);
         nir_builder build = nir_builder_create(impl);
         build.cursor = nir_after_block(nir_start_block(impl));
         clone_sparse_sum(&build, holes, 71);
         nir_def *condition = nir_imm_bool(&build, true);
         for (unsigned i = 0; i < count; ++i) {
            nir_if *nif = nir_push_if(&build, condition);
            nir_push_else(&build, nif);
            nir_pop_if(&build, nif);
         }
         cases.push_back({impl, impl->ssa_alloc, count});
      }
   }
   nir_validate_shader(b->shader, "block-context clone source");
   nir_shader *copy = nir_shader_clone(nullptr, b->shader);
   nir_validate_shader(copy, "block-context clone result");
   nir_validate_ssa_dominance(copy, "block-context clone result");
   for (const block_case &source : cases) {
      nir_function_impl *impl = clone_find_impl(copy, source.impl->function->name);
      ASSERT_NE(impl, nullptr);
      std::vector<nir_block *> old_blocks, new_blocks;
      nir_foreach_block(block, source.impl)
         old_blocks.push_back(block);
      nir_foreach_block(block, impl)
         new_blocks.push_back(block);
      ASSERT_EQ(old_blocks.size(), new_blocks.size());
      EXPECT_EQ(old_blocks.size(), 1u + 3u * source.if_count);
      unsigned conditions = 0;
      nir_def *new_condition = nullptr;
      for (unsigned i = 0; i < old_blocks.size(); ++i) {
         for (unsigned edge = 0; edge < 2; ++edge) {
            nir_block *successor = old_blocks[i]->successors[edge];
            if (!successor) {
               EXPECT_EQ(new_blocks[i]->successors[edge], nullptr);
            } else if (successor == source.impl->end_block) {
               EXPECT_EQ(new_blocks[i]->successors[edge], impl->end_block);
            } else {
               auto found = std::find(old_blocks.begin(), old_blocks.end(), successor);
               ASSERT_NE(found, old_blocks.end());
               EXPECT_EQ(new_blocks[i]->successors[edge],
                         new_blocks[found - old_blocks.begin()]);
            }
         }
         nir_cf_node *old_next = nir_cf_node_next(&old_blocks[i]->cf_node);
         if (old_next && old_next->type == nir_cf_node_if) {
            nir_cf_node *new_next = nir_cf_node_next(&new_blocks[i]->cf_node);
            ASSERT_NE(new_next, nullptr);
            ASSERT_EQ(static_cast<unsigned>(new_next->type),
                      static_cast<unsigned>(nir_cf_node_if));
            nir_if *old_if = nir_cf_node_as_if(old_next);
            nir_if *new_if = nir_cf_node_as_if(new_next);
            EXPECT_NE(new_if->condition.ssa, old_if->condition.ssa);
            ASSERT_TRUE(nir_src_is_const(new_if->condition));
            EXPECT_TRUE(nir_def_as_load_const(new_if->condition.ssa)->value[0].b);
            if (!new_condition)
               new_condition = new_if->condition.ssa;
            EXPECT_EQ(new_if->condition.ssa, new_condition);
            ++conditions;
         }
      }
      EXPECT_EQ(conditions, source.if_count);
      EXPECT_EQ(source.impl->ssa_alloc, source.capacity);
   }
   ralloc_free(copy);
}

static void
collect_cf_tree_blocks(struct exec_list *list, std::vector<nir_block *> &blocks)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         blocks.push_back(nir_cf_node_as_block(node));
         break;
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         collect_cf_tree_blocks(&nif->then_list, blocks);
         collect_cf_tree_blocks(&nif->else_list, blocks);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         collect_cf_tree_blocks(&loop->body, blocks);
         collect_cf_tree_blocks(&loop->continue_list, blocks);
         break;
      }
      default:
         ADD_FAILURE();
         break;
      }
   }
}

static void
build_cf_tree_case(nir_builder *build, unsigned depth, unsigned style)
{
   if (!depth)
      return;

   nir_if *nif = nir_push_if(build, nir_imm_true(build));
   build_cf_tree_case(build, depth - 1, style);
   nir_push_else(build, nif);
   if (style & 1)
      build_cf_tree_case(build, depth - 1, style);
   nir_pop_if(build, nif);

   if (style & 2) {
      nir_loop *loop = nir_push_loop(build);
      if (style & 4)
         nir_loop_add_continue_construct(loop);
      build_cf_tree_case(build, depth - 1, style);
      nir_if *exit = nir_push_if(build, nir_imm_true(build));
      nir_jump(build, nir_jump_break);
      nir_push_else(build, exit);
      if (style & 4)
         nir_jump(build, nir_jump_continue);
      nir_pop_if(build, exit);
      if (style & 4) {
         nir_push_continue(build, loop);
         build_cf_tree_case(build, depth - 1, style & 1);
      }
      nir_pop_loop(build, loop);
   }
}

TEST_F(nir_core_test, cf_tree_next_matches_recursive_lists)
{
   EXPECT_EQ(nir_block_cf_tree_next(nullptr), nullptr);
   for (unsigned depth = 0; depth <= 3; ++depth) {
      for (unsigned style : {0u, 1u, 2u, 3u, 6u, 7u}) {
         SCOPED_TRACE(depth);
         SCOPED_TRACE(style);
         nir_shader_compiler_options options = {};
         nir_builder current = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &options, "cf tree traversal");
         ralloc_steal(b->shader, current.shader);
         nir_imm_int(&current, 42);
         build_cf_tree_case(&current, depth, style);
         nir_validate_shader(current.shader, "before cf tree traversal");

         auto check = [&](struct exec_list *list) {
            std::vector<nir_block *> expected;
            collect_cf_tree_blocks(list, expected);
            ASSERT_FALSE(expected.empty());
            for (unsigned i = 0; i < expected.size(); ++i) {
               SCOPED_TRACE(i);
               EXPECT_EQ(nir_block_cf_tree_next(expected[i]),
                         i + 1 < expected.size() ? expected[i + 1] : nullptr);
            }
         };
         check(&current.impl->body);
         nir_cf_list extracted;
         nir_cursor gap = nir_cf_extract(
            &extracted, nir_before_cf_list(&current.impl->body),
            nir_after_cf_list(&current.impl->body));
         check(&extracted.list);
         nir_cf_reinsert(&extracted, gap);
         nir_validate_shader(current.shader, "after cf tree reinsertion");
         check(&current.impl->body);
      }
   }
}

}
