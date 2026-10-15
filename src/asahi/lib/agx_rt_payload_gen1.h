/* SPDX-License-Identifier: MIT */
#pragma once

#include <math.h>
#include "agx_rt_bvh.h"

/* G17P/RIA2 RayCore segment, decoded from native 25G83
 * agx.intersection_query.next (payload_accessor_gen1::init_core).
 * This is the 32-byte core segment only. Query, mask, token, and continuation
 * segments have separate native layouts and must be initialized by the caller.
 * The caller supplies the already winding-adjusted native cull mode (0..2).
 * This helper does not enable Vulkan ray query or provide a traversal ABI.
 */
static inline bool
agx_rt_payload_gen1_core(void *out, const float origin[3],
                        const float direction[3], float t_max,
                        const float bounds_min[3], const float bounds_max[3],
                        unsigned cull_mode, bool flag16)
{
   if (!out || !origin || !direction || !bounds_min || !bounds_max ||
       cull_mode > 2 || !agx_rt_bvh_is_finite(t_max) || t_max < 0)
      return false;

   float far[3];
   for (unsigned i = 0; i < 3; ++i) {
      if (!agx_rt_bvh_is_finite(origin[i]) ||
          !agx_rt_bvh_is_finite(direction[i]) ||
          !agx_rt_bvh_is_finite(bounds_min[i]) ||
          !agx_rt_bvh_is_finite(bounds_max[i]) || bounds_min[i] > bounds_max[i])
         return false;
      float lo = origin[i] - bounds_min[i];
      float hi = origin[i] - bounds_max[i];
      lo *= lo > 0 ? 0x1.fffffcp-1f : 0x1.000002p0f;
      hi *= hi > 0 ? 0x1.000002p0f : 0x1.fffffcp-1f;
      far[i] = fmaxf(fabsf(lo), fabsf(hi));
   }

   unsigned major = fabsf(direction[0]) < fabsf(direction[1]) ? 1 : 0;
   if (fabsf(direction[major]) < fabsf(direction[2]))
      major = 2;
   if (direction[major] == 0)
      return false;
   unsigned other0 = (major + 1) % 3, other1 = (major + 2) % 3;
   float epsilon = (far[major] + fmaxf(far[other0], far[other1])) * 0x1.4p-22f;
   uint32_t quantized =
      ((agx_rt_bvh_bitcast_f32_to_u32(epsilon) + 131071u) >> 17) & 0x3fff;
   /* Native permutation table {0x10002, 0x20000, 1}. */
   static const uint32_t permutations[] = {0x10002, 0x20000, 1};
   unsigned high = permutations[major] >> 16;
   unsigned low = permutations[major] & 0xffff;
   unsigned sx = direction[major] < 0 ? low : high;
   unsigned sy = direction[major] < 0 ? high : low;
   uint32_t control = major | (quantized << 2) | (flag16 ? 1u << 16 : 0) |
                      (cull_mode << 18) | (sx << 20) | (sy << 22);

   uint8_t bytes[32];
   for (unsigned i = 0; i < 3; ++i) {
      agx_store_le32(bytes + 8 * i,
                    agx_rt_bvh_bitcast_f32_to_u32(origin[i]));
      agx_store_le32(bytes + 8 * i + 4,
                    agx_rt_bvh_bitcast_f32_to_u32(direction[i]));
   }
   agx_store_le32(bytes + 24, agx_rt_bvh_bitcast_f32_to_u32(t_max));
   agx_store_le32(bytes + 28, control);
   memcpy(out, bytes, sizeof(bytes));
   return true;
}
