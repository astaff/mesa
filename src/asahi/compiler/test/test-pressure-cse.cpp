/* SPDX-License-Identifier: MIT */
#include <vector>
#include <gtest/gtest.h>
#include "agx_test.h"

static unsigned
count_uniform_copies(agx_context *ctx, unsigned uniform)
{
   unsigned count = 0;
   agx_foreach_instr_global(ctx, I) {
      count += I->op == AGX_OPCODE_MOV && I->src[0].type == AGX_INDEX_UNIFORM &&
               I->src[0].value == uniform;
   }
   return count;
}

TEST(PressureCSE, SharesOverlappingCopies)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_index a = agx_mov(b, agx_uniform(8, AGX_SIZE_32));
   agx_index c = agx_mov(b, agx_uniform(8, AGX_SIZE_32));
   agx_instr *use_a = agx_unit_test(b, a);
   agx_instr *use_c = agx_unit_test(b, c);
   agx_opt_cse_pressure(b->shader);
   agx_dce(b->shader, true);
   EXPECT_EQ(count_uniform_copies(b->shader, 8), 1u);
   EXPECT_TRUE(agx_is_equiv(use_a->src[0], use_c->src[0]));
   ralloc_free(mem);
}

TEST(PressureCSE, PreservesCopiesAcrossFullPressureGap)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));
   agx_index x = agx_mov(b, agx_uniform(10, AGX_SIZE_32));
   agx_index y = agx_mov(b, agx_uniform(12, AGX_SIZE_32));
   agx_unit_test(b, x);
   agx_unit_test(b, y);
   agx_index next = agx_mov(b, agx_uniform(8, AGX_SIZE_32));
   agx_index last = agx_mov(b, agx_uniform(8, AGX_SIZE_32));
   agx_instr *use_next = agx_unit_test(b, next);
   agx_instr *use_last = agx_unit_test(b, last);
   agx_opt_cse_pressure(b->shader);
   agx_dce(b->shader, true);
   EXPECT_EQ(count_uniform_copies(b->shader, 8), 2u);
   EXPECT_TRUE(agx_is_equiv(use_next->src[0], use_last->src[0]));
   ralloc_free(mem);
}

TEST(PressureCSE, DoesNotExtendLifetimesIntoPressureHeadroom)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));
   agx_unit_test(b, agx_mov(b, agx_uniform(10, AGX_SIZE_32)));
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));
   agx_index x = agx_mov(b, agx_uniform(12, AGX_SIZE_32));
   agx_index y = agx_mov(b, agx_uniform(14, AGX_SIZE_32));
   agx_index z = agx_mov(b, agx_uniform(16, AGX_SIZE_32));
   agx_unit_test(b, x);
   agx_unit_test(b, y);
   agx_unit_test(b, z);
   agx_opt_cse_pressure(b->shader);
   agx_dce(b->shader, true);
   EXPECT_EQ(count_uniform_copies(b->shader, 8), 2u);
   ralloc_free(mem);
}

TEST(PressureCSE, PreservesIndependentBlockDefinitions)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));
   agx_block *block = agx_test_block(b->shader);
   b->cursor = agx_after_block(block);
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));
   agx_opt_cse_pressure(b->shader);
   agx_dce(b->shader, true);
   EXPECT_EQ(count_uniform_copies(b->shader, 8), 2u);
   ralloc_free(mem);
}

static unsigned
peak_pressure(agx_context *ctx)
{
   std::vector<unsigned> first(ctx->alloc, ~0u), last(ctx->alloc, 0),
      width(ctx->alloc, 0);
   unsigned position = 0;
   agx_foreach_instr_global(ctx, I) {
      ++position;
      agx_foreach_ssa_dest(I, d) {
         first[I->dest[d].value] = position;
         width[I->dest[d].value] = agx_index_size_16(I->dest[d]);
      }
      agx_foreach_ssa_src(I, s)
         last[I->src[s].value] = position;
   }
   unsigned peak = 0;
   for (unsigned p = 1; p <= position; ++p) {
      unsigned live = 0;
      for (unsigned v = 0; v < ctx->alloc; ++v)
         if (first[v] <= p && p <= last[v])
            live += width[v];
      peak = MAX2(peak, live);
   }
   return peak;
}

static std::vector<unsigned>
observe_uniforms(agx_context *ctx)
{
   std::vector<unsigned> values(ctx->alloc, ~0u), observed;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_MOV) {
         EXPECT_EQ(I->src[0].type, AGX_INDEX_UNIFORM);
         values[I->dest[0].value] = I->src[0].value;
      } else if (I->op == AGX_OPCODE_UNIT_TEST) {
         observed.push_back(values[I->src[0].value]);
      }
   }
   return observed;
}

TEST(PressureCSE, RandomizedValuesAndPressureInvariant)
{
   for (unsigned seed = 1; seed <= 128; ++seed) {
      SCOPED_TRACE(seed);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      std::vector<agx_index> values;
      unsigned state = seed;
      auto random = [&state]() {
         state = state * 1664525u + 1013904223u;
         return state;
      };
      for (unsigned i = 0; i < 64; ++i) {
         auto size = random() & 1 ? AGX_SIZE_16 : AGX_SIZE_32;
         values.push_back(agx_mov(b, agx_uniform(2 * (random() % 8), size)));
         if (random() & 3)
            agx_unit_test(b, values[random() % values.size()]);
      }
      for (unsigned i = 0; i < 32; ++i)
         agx_unit_test(b, values[random() % values.size()]);
      auto expected = observe_uniforms(b->shader);
      unsigned pressure = peak_pressure(b->shader);
      agx_opt_cse_pressure(b->shader);
      agx_dce(b->shader, true);
      EXPECT_EQ(observe_uniforms(b->shader), expected);
      EXPECT_LE(peak_pressure(b->shader), pressure);
      ralloc_free(mem);
   }
}
