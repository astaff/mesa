/* SPDX-License-Identifier: MIT */
#include "agx_rt_payload_gen1.h"
#include <assert.h>
#include <stdio.h>

struct gen1_core_case {
   float origin[3], direction[3], minimum[3], maximum[3];
   unsigned cull_mode;
   uint32_t expected[8];
};
#include "rt-payload-gen1-native.h"

int main(void)
{
   for (unsigned i = 0; i < sizeof(native_gen1_core) / sizeof(*native_gen1_core); ++i) {
      const struct gen1_core_case *c = &native_gen1_core[i];
      uint8_t actual[32];
      assert(agx_rt_payload_gen1_core(actual, c->origin, c->direction, 100,
                                     c->minimum, c->maximum, c->cull_mode, false));
      assert(memcmp(actual, c->expected, sizeof(actual)) == 0);
   }
   uint8_t out[32], original[32];
   memset(out, 0xa5, sizeof(out));
   memcpy(original, out, sizeof(out));
   const struct gen1_core_case *c = &native_gen1_core[0];
   const float zero[3] = {0, 0, 0};
   assert(!agx_rt_payload_gen1_core(out, c->origin, zero, 100,
                                   c->minimum, c->maximum, 0, false));
   assert(!agx_rt_payload_gen1_core(out, c->origin, c->direction, 100,
                                   c->minimum, c->maximum, 3, false));
   assert(memcmp(out, original, sizeof(out)) == 0);
   puts("128 native Gen1 RayCore vectors and transactional rejection passed");
   return 0;
}
