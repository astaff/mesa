/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"

TEST(SpillFallback, EdgeStoresReportScratchDuringRematerialization)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      SCOPED_TRACE(mode);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.has_scratch = true;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->any_cf = true;
      b->shader->is_preamble = true;
      agx_block *entry = agx_start_block(b->shader);
      agx_block *left = agx_test_block(b->shader);
      agx_block *right = agx_test_block(b->shader);
      agx_block *join = agx_test_block(b->shader);
      agx_block_add_successor(entry, left);
      agx_block_add_successor(entry, right);
      agx_block_add_successor(left, join);
      agx_block_add_successor(right, join);
      agx_index incoming[2];
      unsigned side = 0;
      for (agx_block *block : {left, right}) {
         b->cursor = agx_after_block(block);
         if (mode == 0)
            incoming[side] = agx_mov(b, agx_uniform(2 * side, AGX_SIZE_32));
         else if (mode == 1)
            incoming[side] = agx_mov_imm(b, 32, 13 + side);
         else if (mode == 2) {
            incoming[side] = agx_immediate(13 + side);
         } else
            incoming[side] = agx_uniform(2 * side, AGX_SIZE_32);
         ++side;
      }
      b->cursor = agx_after_block(join);
      agx_index values[16];
      for (agx_index &value : values) {
         value = agx_temp(b->shader, AGX_SIZE_32);
         agx_instr *phi = agx_phi_to(b, value, 2);
         phi->src[0] = incoming[0];
         phi->src[1] = incoming[1];
      }
      for (agx_index value : values)
         agx_unit_test(b, value);
      agx_compute_liveness(b->shader);
      agx_spill(b->shader, 24, true);
      EXPECT_TRUE(b->shader->has_spill_pcopy_reserved);
      EXPECT_TRUE(b->shader->remat_fallback_spilled);
      unsigned stores = 0;
      agx_foreach_instr_global(b->shader, I) {
         stores += I->op == AGX_OPCODE_MOV && I->dest[0].memory;
      }
      EXPECT_GT(stores, 0u);
      ralloc_free(mem);
   }
}
