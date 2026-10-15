#ifndef AGX_RT_COMPUTE_GEN1_H
#define AGX_RT_COMPUTE_GEN1_H

#include <stdbool.h>
#include <stdint.h>

static inline bool
agx_rt_compute_gen1_context_local(uint32_t context[28], uint32_t x,
                                 uint32_t y, uint32_t z)
{
   if (!context || !x || !y || !z || x > 1024 || y > 1024 || z > 1024)
      return false;
   const uint64_t xy = (uint64_t)x * y;
   const uint64_t volume = xy * z;
   if (volume > 1024)
      return false;
   uint32_t divisor = 0xffff0000;
   if (xy > 1) {
      uint32_t shift = 0;
      for (uint64_t n = xy - 1; n > 1; n >>= 1)
         ++shift;
      const uint64_t multiplier =
         (((UINT64_C(1) << (shift + 16)) + xy - 1) / xy) - 1;
      divisor = ((uint32_t)multiplier << 16) | shift;
   }
   context[9] = x;
   context[10] = y;
   context[11] = z;
   context[26] = (uint32_t)volume | ((uint32_t)xy << 16);
   context[27] = divisor;
   return true;
}

#endif
