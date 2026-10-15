/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_pack.h"
#include "agx_vdm_mesh_core.h"
#include <cstring>
#include <array>

#include <gtest/gtest.h>

const struct {
   float f;
   uint16_t encoded;
   bool inexact;
} lod_cases[] = {
   /* Lower bound clamp */
   {-INFINITY, 0x00, true},
   {-0.1, 0x00, true},
   {-0.0, 0x00, true},

   /* Exact bounds */
   {0.0, 0x00},
   {14.0, 0x380},

   /* Upper bound clamp */
   {14.1, 0x380, true},
   {18.1, 0x380, true},
   {INFINITY, 0x380, true},
};

const struct {
   uint32_t group_size;
   uint32_t length;
   uint32_t value;
   uint32_t encoded;
} group_cases[] = {
   /* Groups of 16 in a 4-bit word */
   {16, 4, 0, 0x1},   {16, 4, 1, 0x1},   {16, 4, 16, 0x1},  {16, 4, 17, 0x2},
   {16, 4, 31, 0x2},  {16, 4, 32, 0x2},  {16, 4, 33, 0x3},  {16, 4, 239, 0xF},
   {16, 4, 240, 0xF}, {16, 4, 241, 0x0}, {16, 4, 255, 0x0}, {16, 4, 256, 0x0},
};

TEST(LODClamp, Encode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(lod_cases); ++i)
      ASSERT_EQ(__gen_pack_lod(lod_cases[i].f, 0, 9), lod_cases[i].encoded);
}

TEST(LODClamp, Decode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(lod_cases); ++i) {
      if (lod_cases[i].inexact)
         continue;

      uint32_t cl;
      memcpy(&cl, &lod_cases[i].encoded, sizeof(lod_cases[i].encoded));

      ASSERT_EQ(__gen_unpack_lod(&cl, 0, 10), lod_cases[i].f);
   }
}

TEST(Groups, Encode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(group_cases); ++i) {
      ASSERT_EQ(__gen_to_groups(group_cases[i].value, group_cases[i].group_size,
                                group_cases[i].length),
                group_cases[i].encoded);
   }
}

TEST(Groups, Decode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(group_cases); ++i) {
      unsigned expected =
         ALIGN_POT(group_cases[i].value, group_cases[i].group_size);

      /* Clamp to minimum encodable */
      if (group_cases[i].value == 0)
         expected = group_cases[i].group_size;

      ASSERT_EQ(
         __gen_from_groups(group_cases[i].encoded, group_cases[i].group_size,
                           group_cases[i].length),
         expected);
   }
}


TEST(M5MeshLimits, MatchesNativePackets)
{
   struct { uint32_t vertices, primitives, words, prefix, prim_limit, vert_limit; } cases[] = {
      {6, 2, 4, 0x504, 0x1, 0x9},
      {6, 2, 5, 0x605, 0x1, 0xb},
      {192, 64, 4, 0x504, 0x3003f, 0xc1},
   };
   for (const auto &c : cases) {
      agx_vdm_mesh_state_info info = {};
      ASSERT_TRUE(agx_vdm_m5_mesh_set_output_limits(&info, c.vertices, c.primitives, c.words));
      EXPECT_EQ(info.raw_prefix[0], c.prefix);
      EXPECT_EQ(info.raw_tail[1], c.prim_limit);
      EXPECT_EQ(info.raw_tail[4], c.vert_limit);
   }
}

TEST(M5MeshLimits, RejectsInvalidLimitsWithoutMutation)
{
   for (const auto &c : {std::array<uint32_t, 3>{0, 1, 4}, {1, 0, 4},
                        {257, 1, 4}, {1, 257, 4}, {1, 1, 3}, {1, 1, 256}}) {
      agx_vdm_mesh_state_info info;
      memset(&info, 0x5a, sizeof(info));
      const auto before = info;
      EXPECT_FALSE(agx_vdm_m5_mesh_set_output_limits(&info, c[0], c[1], c[2]));
      EXPECT_EQ(memcmp(&info, &before, sizeof(info)), 0);
   }
}
