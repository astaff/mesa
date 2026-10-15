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
}
#include <vector>

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

}
