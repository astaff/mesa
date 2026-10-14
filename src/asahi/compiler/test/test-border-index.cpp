#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "asahi/vulkan/hk_nir_border_index.h"
}

TEST(BorderIndex, MatchesOnlyRelatedImmutableDescriptorFields)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "border descriptor provenance");
   nir_def *base = nir_load_global_invocation_id(&b, 32);
   nir_def *ptr = nir_u2u64(&b, nir_channel(&b, base, 0));
   nir_def *other = nir_u2u64(&b, nir_channel(&b, base, 1));
   auto load = [&](nir_def *p, unsigned offset, unsigned bits, bool commute,
                   enum gl_access_qualifier access) {
      nir_def *constant = nir_imm_int64(&b, offset);
      nir_def *addr = commute ? nir_iadd(&b, constant, p)
                             : nir_iadd(&b, p, constant);
      return nir_load_global_constant(&b, 1, bits, addr, .access = access,
                                      .align_mul = 2);
   };
   auto original = load(ptr, 690, 16, false, ACCESS_CAN_SPECULATE);
   auto unbiased = load(ptr, 682, 16, false, ACCESS_CAN_SPECULATE);
   EXPECT_TRUE(hk_nir_border_indices_same_descriptor(
      nir_get_scalar(original, 0), nir_get_scalar(unbiased, 0), 8));
   auto commuted = load(ptr, 682, 16, true, ACCESS_CAN_SPECULATE);
   EXPECT_TRUE(hk_nir_border_indices_same_descriptor(
      nir_get_scalar(original, 0), nir_get_scalar(commuted, 0), 8));
   auto wrong_base = load(other, 682, 16, false, ACCESS_CAN_SPECULATE);
   auto wrong_field = load(ptr, 680, 16, false, ACCESS_CAN_SPECULATE);
   auto wrong_width = load(ptr, 682, 32, false, ACCESS_CAN_SPECULATE);
   auto volatile_load = load(ptr, 682, 16, false, ACCESS_VOLATILE);
   for (nir_def *invalid : {wrong_base, wrong_field, wrong_width, volatile_load}) {
      EXPECT_FALSE(hk_nir_border_indices_same_descriptor(
         nir_get_scalar(original, 0), nir_get_scalar(invalid, 0), 8));
   }
   nir_def *variable = nir_load_global_constant(
      &b, 1, 16, nir_iadd(&b, ptr, other), .align_mul = 2);
   EXPECT_FALSE(hk_nir_border_indices_same_descriptor(
      nir_get_scalar(original, 0), nir_get_scalar(variable, 0), 8));
   nir_validate_shader(b.shader, "border descriptor provenance");
   ralloc_free(b.shader);
}
