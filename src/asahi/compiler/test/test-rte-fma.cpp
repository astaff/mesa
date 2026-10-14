/* SPDX-License-Identifier: MIT */
#include <vector>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "asahi/layout/layout.h"
#include "agx_compile.h"
}

static std::vector<uint8_t>
compile_native_fma(unsigned mode, unsigned components)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "native fused FP32 operation");
   b.shader->info.float_controls_execution_mode = mode;
   b.fp_math_ctrl = nir_fp_preserve_sz_inf_nan;
   nir_def *x = agx_test_load_preamble(&b, components, 32, 4);
   nir_def *y = agx_test_load_preamble(&b, components, 32, 12);
   nir_def *z = agx_test_load_preamble(&b, components, 32, 20);
   nir_def *result = nir_ffma(&b, x, y, z);
   agx_test_store_local_pixel_agx(
      &b, nir_pad_vector(&b, result, 4), nir_imm_intN_t(&b, 0xff, 16),
      nir_undef(&b, 2, 16), 0, BITFIELD_MASK(components),
      (enum pipe_format)AIL_ISA_FORMAT_I32);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.no_preamble = true;
   key.reserved_preamble = 32;
   key.fs.nr_samples = 1;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   EXPECT_TRUE(compiled);
   std::vector<uint8_t> binary;
   if (compiled) {
      const uint8_t *main = (const uint8_t *)part.binary + part.info.main_offset;
      binary.assign(main, main + part.info.main_size);
   }
   free(part.binary);
   ralloc_free(b.shader);
   return binary;
}

TEST(NativeFMA, ExplicitRTEKeepsTheNativeFusedProgram)
{
   for (unsigned components = 1; components <= 4; ++components) {
      SCOPED_TRACE(components);
      auto plain = compile_native_fma(0, components);
      auto rte = compile_native_fma(FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32,
                                    components);
      EXPECT_FALSE(plain.empty());
      EXPECT_EQ(plain, rte);
   }
}

TEST(NativeFMA, FlushModesKeepTheNativeFusedProgram)
{
   for (unsigned components = 1; components <= 4; ++components) {
      SCOPED_TRACE(components);
      auto plain = compile_native_fma(0, components);
      auto ftz = compile_native_fma(FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32,
                                    components);
      auto rte_ftz = compile_native_fma(
         FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32 |
         FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32, components);
      EXPECT_FALSE(plain.empty());
      EXPECT_EQ(plain, ftz);
      EXPECT_EQ(plain, rte_ftz);
   }
}
