/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_nir.h"
}

static void
check_offset(unsigned bits, unsigned add, unsigned base, uint32_t mask,
             bool store, bool expected)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "shared-byte-offset");
   nir_def *v = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 4);
   v = nir_iand_imm(&b, v, mask);
   nir_def *sum = nir_iadd_imm(&b, v, add);
   nir_def *index = nir_u2u16(&b, nir_ushr_imm(&b, sum, util_logbase2(bits / 8)));
   nir_intrinsic_instr *intr;
   if (store) {
      intr = nir_store_shared(&b, nir_imm_intN_t(&b, 0, bits), index,
                              .base = base, .align_mul = bits / 8);
   } else {
      nir_def *value = nir_load_shared(&b, 1, bits, index, .base = base,
                                        .align_mul = bits / 8);
      intr = nir_def_as_intrinsic(value);
   }
   EXPECT_EQ(agx_nir_opt_shared_byte_offsets(b.shader), expected);
   EXPECT_EQ(nir_intrinsic_base(intr), base + (expected ? add : 0));
   EXPECT_EQ(nir_get_io_offset_src(intr)->ssa == index, !expected);
   nir_validate_shader(b.shader, "shared-byte-offset result");
   ralloc_free(b.shader);
}

TEST(SharedByteOffsets, Load32) { check_offset(32, 12, 0, 4095, false, true); }
TEST(SharedByteOffsets, Store32) { check_offset(32, 56, 0, 4095, true, true); }
TEST(SharedByteOffsets, Load16) { check_offset(16, 14, 0, 4095, false, true); }
TEST(SharedByteOffsets, ExistingBase) { check_offset(32, 12, 16, 4095, false, true); }
TEST(SharedByteOffsets, ByteOffsetTooLarge) { check_offset(32, 256, 0, 4095, false, false); }
TEST(SharedByteOffsets, BaseOverflow) { check_offset(32, 16, 240, 4095, false, false); }
TEST(SharedByteOffsets, Misaligned) { check_offset(32, 3, 0, 4095, false, false); }
TEST(SharedByteOffsets, AdditionWrap) { check_offset(32, 12, 0, UINT32_MAX, false, false); }
TEST(SharedByteOffsets, NarrowingWrap) { check_offset(32, 12, 0, 262143, false, false); }
TEST(SharedByteOffsets, ExactBoundary) { check_offset(32, 12, 0, 262131, false, true); }
