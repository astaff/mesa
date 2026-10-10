/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_nir.h"
}

static unsigned
fadds(nir_shader *s)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            n += instr->type == nir_instr_type_alu &&
                 nir_instr_as_alu(instr)->op == nir_op_fadd;
         }
      }
   }
   return n;
}

static void
constants(bool is_max)
{
   const struct { uint32_t bits; bool flush; } cases[] = {
      {0, false}, {0x80000000u, false},
      {1, true}, {0x80000001u, true},
      {0x007fffffu, true}, {0x807fffffu, true},
      {0x00800000u, false}, {0x80800000u, false},
      {0x3f800000u, false}, {0xbf800000u, false},
      {0x7f800000u, false}, {0xff800000u, false},
      {0x7fc00000u, false}, {0x7f800001u, false},
   };
   for (const auto &test : cases) {
      const uint32_t bits = test.bits;
      SCOPED_TRACE(bits);
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                      &agx_nir_options, "constant-ftz");
      nir_def *a = nir_imm_int(&b, bits);
      nir_def *other = nir_imm_float(&b, 1.0f);
      nir_def *out = is_max ? nir_fmax(&b, a, other) : nir_fmin(&b, a, other);
      nir_intrinsic_instr *store = nir_store_output(&b, out, nir_imm_int(&b, 0));
      nir_io_semantics semantics = {};
      semantics.location = FRAG_RESULT_DATA0;
      nir_intrinsic_set_io_semantics(store, semantics);
      ASSERT_TRUE(agx_nir_lower_fminmax(b.shader));
      EXPECT_EQ(fadds(b.shader), unsigned(test.flush));
      nir_validate_shader(b.shader, "constant-ftz");
      ralloc_free(b.shader);
   }
}
TEST(FminmaxConstantDenorm, DirectConstantMin) { constants(false); }
TEST(FminmaxConstantDenorm, DirectConstantMax) { constants(true); }
