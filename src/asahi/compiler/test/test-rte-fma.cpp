/* SPDX-License-Identifier: MIT */
#include <cmath>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "asahi/layout/layout.h"
#include "agx_nir.h"
}

static unsigned
lower_fma(unsigned slot, uint32_t bits, bool rte = true)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "constant FMA rounding");
   if (rte)
      b.shader->info.float_controls_execution_mode =
         FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32;
   b.fp_math_ctrl = nir_fp_preserve_sz_inf_nan;
   nir_def *coord = nir_load_frag_coord(&b);
   nir_def *src[3] = {nir_channel(&b, coord, 0), nir_channel(&b, coord, 1),
                      nir_channel(&b, coord, 2)};
   src[slot] = nir_imm_int(&b, bits);
   nir_def *value = nir_ffma(&b, src[0], src[1], src[2]);
   agx_test_store_local_pixel_agx(&b, nir_vec4(&b, value, value, value, value),
                                  nir_imm_intN_t(&b, 0xff, 16),
                                  nir_undef(&b, 2, 16), 0, 0xf,
                                  (enum pipe_format)AIL_ISA_FORMAT_I32);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_nir_lower_rte_fma(b.shader);
   nir_validate_shader(b.shader, "RTE FMA repair");
   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
   nir_index_blocks(impl);
   unsigned blocks = impl->num_blocks;
   ralloc_free(b.shader);
   return blocks;
}

TEST(RTEFMA, OmitsUnreachableRepairPaths)
{
   for (unsigned slot : {0u, 1u}) {
      for (uint32_t bits :
           {0u, 0x80000000u, 0x00000001u, 0x007fffffu, 0x3f800000u, 0x40000000u,
            0x7f000000u, 0xbf800000u, 0x7f800000u, 0x7fc00000u}) {
         SCOPED_TRACE(slot);
         SCOPED_TRACE(bits);
         EXPECT_EQ(lower_fma(slot, bits), 1u);
      }
   }
   for (uint32_t bits : {0x0d000000u, 0x0d7fffffu, 0x3f800000u, 0xbf800000u,
                         0x7f800000u, 0x7fc00000u}) {
      SCOPED_TRACE(bits);
      EXPECT_EQ(lower_fma(2, bits), 1u);
   }
}

TEST(RTEFMA, RetainsPossibleBoundaryRepairs)
{
   for (unsigned slot : {0u, 1u}) {
      for (uint32_t bits :
           {0x00800000u, 0x3f000000u, 0x3f7fffffu, 0x3f800001u, 0xbf000000u}) {
         SCOPED_TRACE(slot);
         SCOPED_TRACE(bits);
         EXPECT_GT(lower_fma(slot, bits), 1u);
      }
   }
   for (uint32_t bits : {0u, 0x80000000u, 0x00800000u, 0x0cffffffu}) {
      SCOPED_TRACE(bits);
      EXPECT_GT(lower_fma(2, bits), 1u);
   }
   EXPECT_EQ(lower_fma(2, 0x00800000u, false), 1u);
}

static bool
needs_boundary_repair(float x, float y, float z)
{
   auto flush = [](float f) {
      return (fui(f) & 0x7f800000u) == 0 ? std::copysign(0.0f, f) : f;
   };
   x = flush(x);
   y = flush(y);
   z = flush(z);
   double product = double(x) * double(y);
   double exact = product + double(z);
   double magnitude = std::fabs(exact);
   return magnitude >= std::ldexp(1.0, -126) - std::ldexp(1.0, -150) &&
          magnitude < std::ldexp(1.0, -126);
}

TEST(RTEFMA, ReferenceBoundaryExamples)
{
   EXPECT_TRUE(needs_boundary_repair(uif(0x00800000), uif(0x3f7fffff), 0));
   EXPECT_TRUE(needs_boundary_repair(uif(0x00ffffff), 0.5f, 0));
   EXPECT_TRUE(needs_boundary_repair(uif(0x00800000), -std::ldexp(1.0f, -24),
                                     uif(0x00800000)));
   EXPECT_EQ(fui(std::fma(uif(0x00800000), uif(0x3f7fffff), 0.0f)),
             0x00800000u);
}

TEST(RTEFMA, LargeConstantAddendsCannotReachBoundary)
{
   uint32_t state = 1;
   auto random = [&state]() {
      state = state * 1664525u + 1013904223u;
      return state;
   };
   for (unsigned exponent = 26; exponent < 255; ++exponent) {
      for (unsigned i = 0; i < 1024; ++i) {
         float z = uif((exponent << 23) | (random() & 0x807fffffu));
         float x = uif(((1 + random() % 254) << 23) | (random() & 0x807fffffu));
         float center = float(-double(z) / double(x));
         if (!std::isfinite(center))
            continue;
         float y = center;
         for (unsigned adjacent = 0; adjacent < 5; ++adjacent) {
            if (adjacent == 1)
               y = std::nextafter(center, -INFINITY);
            else if (adjacent == 3)
               y = std::nextafter(center, INFINITY);
            else if (adjacent)
               y = std::nextafter(y, adjacent < 3 ? -INFINITY : INFINITY);
            ASSERT_FALSE(needs_boundary_repair(x, y, z));
         }
      }
   }
}
