/* SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include "agx_rt_blas_gen3.h"
#include "rt-envelope-native.h"

static unsigned checks;
static void
require(bool value, const char *label)
{
   checks++;
   if (!value) {
      fprintf(stderr, "FAIL: %s\n", label);
      exit(1);
   }
}

int
main(void)
{
   uint8_t bytes[8192], want[8192], saved[8192];
   const uint64_t native_va = UINT64_C(0x10000058400);
   for (unsigned i = 0;
        i < sizeof(native_envelopes) / sizeof(native_envelopes[0]); ++i) {
      const unsigned tail = native_envelopes[i].end_slot * 64;
      const unsigned size = tail + 184;
      memset(bytes, 0x5a, sizeof(bytes));
      memcpy(saved, bytes, sizeof(bytes));
      memcpy(want, saved, sizeof(want));
      memcpy(want + 64, native_envelopes[i].header, 128);
      memcpy(want + tail, native_envelopes[i].descriptor, 184);
      require(agx_rt_blas_gen3_envelope(
                 bytes, sizeof(bytes), native_va, native_envelopes[i].end_slot,
                 native_envelopes[i].count, native_envelopes[i].nodes,
                 native_envelopes[i].depth, native_envelopes[i].lo,
                 native_envelopes[i].hi, native_envelopes[i].opaque),
              "native envelope builds");
      require(!memcmp(bytes, want, sizeof(bytes)),
              "every envelope byte matches native, graph/guards untouched");
      memcpy(bytes, saved, sizeof(bytes));
      for (unsigned limit = 0; limit < size; ++limit) {
         require(!agx_rt_blas_gen3_envelope(
                    bytes, limit, native_va, native_envelopes[i].end_slot,
                    native_envelopes[i].count, native_envelopes[i].nodes,
                    native_envelopes[i].depth, native_envelopes[i].lo,
                    native_envelopes[i].hi, native_envelopes[i].opaque),
                 "truncated envelope refused");
         require(!memcmp(bytes, saved, sizeof(bytes)),
                 "truncated envelope is transactional");
      }
      const uint64_t invalid_va[] = {0, native_va + 4, UINT64_C(1) << 48,
                                     (UINT64_C(1) << 48) - 1024};
      for (unsigned j = 0; j < sizeof(invalid_va) / sizeof(invalid_va[0]);
           ++j) {
         if (size <= 1024 && j == 3)
            continue;
         require(!agx_rt_blas_gen3_envelope(
                    bytes, sizeof(bytes), invalid_va[j],
                    native_envelopes[i].end_slot, native_envelopes[i].count,
                    native_envelopes[i].nodes, native_envelopes[i].depth,
                    native_envelopes[i].lo, native_envelopes[i].hi,
                    native_envelopes[i].opaque),
                 "invalid address refused");
         require(!memcmp(bytes, saved, sizeof(bytes)),
                 "invalid address is transactional");
      }
      float bad[3] = {0, 0, 0};
      require(!agx_rt_blas_gen3_envelope(
                 bytes, sizeof(bytes), native_va, native_envelopes[i].end_slot,
                 native_envelopes[i].count, native_envelopes[i].nodes,
                 native_envelopes[i].depth, bad, bad, 0),
              "zero-area envelope refused");
      bad[0] = agx_rt_bvh_bitcast_u32_to_f32(0x7fc00000);
      require(!agx_rt_blas_gen3_envelope(
                 bytes, sizeof(bytes), native_va, native_envelopes[i].end_slot,
                 native_envelopes[i].count, native_envelopes[i].nodes,
                 native_envelopes[i].depth, bad, native_envelopes[i].hi, 0),
              "nonfinite envelope refused");
      require(!memcmp(bytes, saved, sizeof(bytes)),
              "invalid bounds are transactional");
      require(agx_rt_blas_gen3_envelope(
                 bytes, sizeof(bytes), native_va + 0x400,
                 native_envelopes[i].end_slot, native_envelopes[i].count,
                 native_envelopes[i].nodes, native_envelopes[i].depth,
                 native_envelopes[i].lo, native_envelopes[i].hi,
                 native_envelopes[i].opaque),
              "rebased envelope builds");
      require(agx_load_le64(bytes + 0x90) == native_va + 0x400,
              "self pointer rebased");
      agx_store_le64(bytes + 0x90, native_va);
      require(!memcmp(bytes, want, sizeof(bytes)),
              "self pointer is only rebased field");
   }
   printf("PASS: %u native envelope checks\n", checks);
   return 0;
}
