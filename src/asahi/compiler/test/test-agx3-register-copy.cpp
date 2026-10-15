/* SPDX-License-Identifier: MIT */
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include "agx3_copy_core.h"
#include "agx3_select_core.h"
#include "agx_test.h"

TEST(AGX3RegisterCopy, MetalGameVectors)
{
   struct vector {
      unsigned dest, src;
      bool dest_cache, src_cache, discard;
      unsigned wait;
      std::array<uint8_t, 4> bytes;
   };
   const vector vectors[] = {
      {6, 20, false, false, true, 0, {0x3b, 0x14, 0x09, 0x00}},
      {20, 0, true, false, false, 0, {0xab, 0x00, 0x21, 0x00}},
      {22, 26, false, false, true, 0, {0xbb, 0x1a, 0x09, 0x00}},
      {20, 12, true, true, false, 0, {0xab, 0x8c, 0x21, 0x00}},
      {20, 12, false, false, false, 0, {0xab, 0x0c, 0x01, 0x00}},
      {22, 68, true, false, true, 8, {0xbb, 0x44, 0x29, 0x80}},
      {20, 12, true, false, true, 32, {0xab, 0x0c, 0x29, 0xc0}},
   };
   for (const auto &v : vectors) {
      uint8_t bytes[AGX3_COPY_MAX_LENGTH] = {};
      uint8_t len = 0;
      ASSERT_TRUE(agx3_copy_encode_reg32(AGX_ISA_ENCODING_AGX3B, v.dest, v.src,
                                         v.dest_cache, v.src_cache, v.discard,
                                         v.wait, bytes, &len));
      ASSERT_EQ(len, v.bytes.size());
      EXPECT_EQ(memcmp(bytes, v.bytes.data(), len), 0);
   }
}

TEST(AGX3RegisterCopy, RejectWithoutWrites)
{
   for (auto enc : {AGX_ISA_ENCODING_AGX2, AGX_ISA_ENCODING_AGX3B,
                    AGX_ISA_ENCODING_AGX3C}) {
      for (unsigned dest : {0u, 1u, 126u, 128u, 254u, 256u}) {
         for (unsigned src : {0u, 1u, 126u, 128u, 254u, 256u}) {
            for (unsigned wait : {0u, 3u, 64u}) {
               for (bool conflict : {false, true}) {
                  if (enc == AGX_ISA_ENCODING_AGX3B && dest < 128 &&
                      src < 256 && !((dest | src) & 1) && wait == 0 &&
                      !conflict)
                     continue;
                  uint8_t bytes[AGX3_COPY_MAX_LENGTH];
                  memset(bytes, 0xa5, sizeof(bytes));
                  uint8_t len = 0x5a;
                  EXPECT_FALSE(agx3_copy_encode_reg32(enc, dest, src, false,
                                                      conflict, conflict, wait,
                                                      bytes, &len));
                  for (auto byte : bytes)
                     EXPECT_EQ(byte, 0xa5);
                  EXPECT_EQ(len, 0x5a);
               }
            }
         }
      }
   }
}

TEST(AGX3RegisterCopy, OperandFieldRoundtrip)
{
   for (unsigned dest = 0; dest < 128; dest += 2) {
      for (unsigned src = 0; src < 256; src += 2) {
         for (unsigned hints = 0; hints < 6; ++hints) {
            for (unsigned wait : {0u, 1u, 2u, 4u, 8u, 16u, 32u}) {
               const bool dest_cache = hints & 1;
               const bool src_cache = (hints >> 1) == 1;
               const bool discard = (hints >> 1) == 2;
               uint8_t bytes[AGX3_COPY_MAX_LENGTH];
               uint8_t len;
               ASSERT_TRUE(agx3_copy_encode_reg32(AGX_ISA_ENCODING_AGX3B, dest,
                                                  src, dest_cache, src_cache,
                                                  discard, wait, bytes, &len));
               ASSERT_EQ(len, 4);
               EXPECT_EQ(((bytes[0] >> 3) & 0x1e) | ((bytes[2] >> 1) & 0x60),
                         dest);
               EXPECT_EQ((bytes[1] & 0x7f) | ((bytes[3] & 0x10) << 3), src);
               EXPECT_EQ((bytes[2] >> 5) & 1, dest_cache);
               EXPECT_EQ(bytes[1] >> 7, src_cache);
               EXPECT_EQ((bytes[2] >> 3) & 1, discard);
               const unsigned selector = (bytes[3] >> 5) & 7;
               EXPECT_EQ(selector, wait ? __builtin_ctz(wait) + 1u : 0u);
            }
         }
      }
   }
}

TEST(AGX3RegisterCopy, BinaryUsesCompactCopyAndKeepsHighFallback)
{
   for (unsigned dest : {6u, 126u, 128u, 190u}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index src = agx_register(20, AGX_SIZE_32);
      src.discard = true;
      agx_bitop_to(b, agx_register(dest, AGX_SIZE_32), src, agx_immediate(0),
                   AGX3_SEL_TT_MOV);
      agx_stop(b);
      util_dynarray bytes;
      util_dynarray_init(&bytes, mem);
      ASSERT_TRUE(agx3_pack_binary(b->shader, &bytes));
      const auto *data = static_cast<const uint8_t *>(bytes.data);
      if (dest < 128) {
         uint8_t expected[AGX3_COPY_MAX_LENGTH];
         uint8_t len;
         ASSERT_TRUE(agx3_copy_encode_reg32(key.dev.encoding, dest, 20, false,
                                            false, true, 0, expected, &len));
         ASSERT_GE(bytes.size, len);
         EXPECT_EQ(memcmp(data, expected, len), 0);
      } else {
         EXPECT_EQ(data[0], 0x9f);
      }
      ralloc_free(mem);
   }
}
