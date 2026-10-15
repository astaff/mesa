/* SPDX-License-Identifier: MIT */
#include "agx_rt_blas_gen3.h"
#include "rt-blas-gen3-native.h"
#include "rt-blas-gen3-graphs.h"

#include <stdio.h>
#include <stdlib.h>

static unsigned checks;
static void
require(bool condition, const char *message)
{
   ++checks;
   if (!condition) {
      fprintf(stderr, "FAIL: %s\n", message);
      exit(1);
   }
}

int main(void)
{
   const uint64_t address = UINT64_C(0x10000058400);
   const float triangle[3][3] = {{-2, -2, 0}, {2, -2, 0}, {0, 2, 0}};
   unsigned char out[520], other[520], sentinel[520];
   memset(out, 0xa5, sizeof(out));
   memcpy(sentinel, out, sizeof(out));
   require(agx_rt_blas_gen3_single_triangle(out, 504, address, triangle),
           "native baseline builds");
   require(!memcmp(out, native_blas_gen3, 504),
           "all 504 bytes match the independent native capture");
   require(!memcmp(out + 504, sentinel + 504, 16), "output tail untouched");
   require(agx_rt_blas_gen3_single_triangle(other, 504, address + 1024, triangle),
           "rebased baseline builds");
   require(agx_load_le64(other + 0x90) == address + 1024,
           "header points at the rebased allocation");
   memcpy(other + 0x90, out + 0x90, 8);
   require(!memcmp(out, other, 504), "self pointer is the only rebased field");

   for (unsigned size = 0; size < 504; ++size) {
      memcpy(out, sentinel, sizeof(out));
      require(!agx_rt_blas_gen3_single_triangle(out, size, address, triangle),
              "truncated destination refused");
      require(!memcmp(out, sentinel, sizeof(out)), "failure is transactional");
   }
   const uint64_t invalid[] = {0, address + 1, address + 512, UINT64_C(1) << 48};
   for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
      require(!agx_rt_blas_gen3_single_triangle(out, 504, invalid[i], triangle),
              "invalid GPU address refused");
      require(!memcmp(out, sentinel, sizeof(out)), "bad address leaves output");
   }
   require(!agx_rt_blas_gen3_single_triangle(NULL, 504, address, triangle),
           "null output refused");
   require(!agx_rt_blas_gen3_single_triangle(out, 504, address, NULL),
           "null input refused");
   const uint32_t nonfinite[] = {0x7f800000, 0xff800000, 0x7fc00000};
   for (unsigned vertex = 0; vertex < 3; ++vertex)
      for (unsigned axis = 0; axis < 3; ++axis)
         for (unsigned i = 0; i < 3; ++i) {
            float v[3][3];
            memcpy(v, triangle, sizeof(v));
            memcpy(&v[vertex][axis], &nonfinite[i], 4);
            require(!agx_rt_blas_gen3_single_triangle(out, 504, address, v),
                    "nonfinite coordinate refused");
            require(!memcmp(out, sentinel, sizeof(out)), "bad vertex leaves output");
         }

   for (unsigned i = 0; i < sizeof(native_small_graphs) /
                             sizeof(native_small_graphs[0]); ++i) {
      const unsigned count = native_small_graphs[i].count;
      const unsigned bytes = native_small_graphs[i].bytes;
      unsigned char root[64], children[656], saved[656];
      memset(children, 0xa5, sizeof(children));
      memcpy(saved, children, sizeof(saved));
      require(agx_rt_blas_gen3_small_graph(root, children, bytes,
                 native_small_graphs[i].vertices, count, 3), "native graph builds");
      require(!memcmp(root, native_small_graphs[i].root, 64),
              "root matches the native GPU byte for byte");
      require(!memcmp(children, native_small_graphs[i].children, bytes),
              "every QTB and geometry byte matches the native GPU");
      require(!memcmp(children + bytes, saved + bytes, sizeof(saved) - bytes),
              "graph leaves the output tail untouched");
      memset(root, 0x5a, sizeof(root));
      memcpy(children, saved, sizeof(children));
      require(!agx_rt_blas_gen3_small_graph(root, children, bytes - 1,
                 native_small_graphs[i].vertices, count, 3), "short graph refused");
      require(!memcmp(children, saved, sizeof(children)), "short graph transactional");
      for (unsigned j = 0; j < sizeof(root); ++j)
         require(root[j] == 0x5a, "failed graph leaves root untouched");
   }

   /* Check the actual traversal bound after serialization, not just the
    * unquantized AABB in the AS header. Include negative and flat axes. */
   for (unsigned scene = 0; scene < 4096; ++scene) {
      float v[3][3];
      for (unsigned vertex = 0; vertex < 3; ++vertex)
         for (unsigned axis = 0; axis < 3; ++axis) {
            int n = (int)((scene * 131 + axis * 17 + vertex * 29) % 1021) - 510;
            v[vertex][axis] = n / 8.0f;
            if (scene % 7 == 0 && axis == 2)
               v[vertex][axis] = -0.125f;
         }
      require(agx_rt_blas_gen3_single_triangle(out, 504, address, v),
              "finite bounded triangle builds");
      struct agx_rt_bvh_gen1_header root;
      float minimum[3], maximum[3];
      require(agx_rt_bvh_gen1_header_unpack(out, &root), "root decodes");
      require(agx_rt_bvh_gen1_child_bounds(&root, 0, minimum, maximum),
              "traversal bound decodes");
      for (unsigned vertex = 0; vertex < 3; ++vertex)
         for (unsigned axis = 0; axis < 3; ++axis)
            require(minimum[axis] <= v[vertex][axis] &&
                    maximum[axis] >= v[vertex][axis],
                    "quantized bound contains every vertex");
   }
   printf("PASS: %u Gen3 BLAS checks\n", checks);
   return 0;
}
