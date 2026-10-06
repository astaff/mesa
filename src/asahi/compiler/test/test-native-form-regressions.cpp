#include "agx_test.h"
#include "agx3_alu_core.h"
#include "agx3_pack.h"
#include "agx_nir.h"
#include <gtest/gtest.h>
#include <vector>

TEST(NativeForms, BoundedUnaryRegistersUnderPressure)
{
   for (agx_opcode op : {AGX_OPCODE_TRUNC, AGX_OPCODE_DFDX, AGX_OPCODE_DFDY}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index live[64];
      for (unsigned i = 0; i < 64; ++i)
         live[i] = agx_preload(b, agx_register(2 * i, AGX_SIZE_32));
      agx_index input = agx_mov_imm(b, 32, 0x40500000);
      agx_index output = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *unary = agx_alloc_instr(b, op, 1, 1);
      unary->src[0] = input;
      unary->dest[0] = output;
      agx_builder_insert(&b->cursor, unary);
      agx_index sum = output;
      for (unsigned i = 0; i < 64; ++i)
         sum = agx_fadd(b, sum, live[i]);
      agx_export(b, sum, 8);
      agx_stop(b);
      agx_ra(b->shader);
      EXPECT_LT(unary->src[0].value, 128u) << op;
      EXPECT_LT(unary->dest[0].value, 128u) << op;
      unsigned high = 0;
      agx_foreach_instr_global(b->shader, I) {
         agx_foreach_dest(I, d)
            high += I->dest[d].type == AGX_INDEX_REGISTER && I->dest[d].value >= 128;
      }
      EXPECT_GT(high, 0u) << "Unrelated live values must retain the upper bank";
      ralloc_free(mem);
   }
}

TEST(NativeForms, NegatedSqrtStepReachesPacker)
{
   for (bool absolute : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index src = agx_register(112, AGX_SIZE_32);
      src.neg = true;
      src.abs = absolute;
      src.cache = true;
      agx_instr *step = agx_alloc_instr(b, AGX_OPCODE_SRSQRT, 1, 1);
      step->src[0] = src;
      step->dest[0] = agx_register(122, AGX_SIZE_32);
      agx_builder_insert(&b->cursor, step);
      agx_stop(b);
      util_dynarray bytes;
      util_dynarray_init(&bytes, mem);
      EXPECT_TRUE(agx3_pack_binary(b->shader, &bytes));
      EXPECT_GT(bytes.size, 4u);
      ralloc_free(mem);
   }
}

TEST(NativeForms, HalfModifiedMultiplyAndSelect)
{
   for (bool select : {false, true}) {
      SCOPED_TRACE(select);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "half modified operands");
      nir_def *input = agx_test_load_preamble(&b, 4, 32, 4);
      nir_def *x = nir_f2f16(&b, nir_channel(&b, input, 0));
      nir_def *result;
      if (select) {
         nir_def *cmp = agx_test_load_preamble(&b, 2, 16, 12);
         x = nir_channel(&b, cmp, 0);
         nir_def *y = nir_channel(&b, cmp, 1);
         result = nir_bcsel(&b, nir_flt(&b, x, nir_fneg(&b, y)),
                           nir_channel(&b, input, 2),
                           nir_channel(&b, input, 3));
      } else {
         result = nir_f2f32(
            &b, nir_fmul(&b, nir_fabs(&b, x), nir_imm_floatN_t(&b, 0.5, 16)));
      }
      agx_test_store_local_pixel_agx(
         &b, nir_vec4(&b, nir_channel(&b, input, 0), result, result, result),
         nir_imm_intN_t(&b, 0xff, 16), nir_undef(&b, 2, 16), 0, 0xf,
         PIPE_FORMAT_R8_UNORM);
      b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.no_preamble = true;
      key.reserved_preamble = 16;
      key.fs.nr_samples = 1;
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_GT(part.info.main_size, 0u);
      free(part.binary);
   }
}
