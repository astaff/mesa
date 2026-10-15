/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef LIBAGX_GEN1_AS_METADATA_H
#define LIBAGX_GEN1_AS_METADATA_H
#ifndef VULKAN
#include <stdint.h>
#endif
#define LIBAGX_GEN1_AS_METADATA_VERSION 1u
#define LIBAGX_GEN1_AS_METADATA_SLOTS 2048u
#define LIBAGX_GEN1_AS_METADATA_STRIDE 64u

/* Driver-owned sidecar, never stored in undocumented native header fields.
 * Each construction/copy publishes its entry on the GPU. Object creation
 * assigns the ID only; CPU recording never changes live build metadata.
 */
struct libagx_gen1_as_metadata {
   uint64_t native_base;
   uint64_t image_bytes;
   uint64_t retained_offset;
   uint64_t retained_bytes;
   uint64_t storage_bytes;
   uint32_t build_flags;
   uint32_t leaf_bound;
   uint32_t node_bound;
   uint32_t geometry_type;
   uint32_t resource_id;
   uint32_t version;
};

#ifndef VULKAN
static inline uint64_t
libagx_gen1_as_compacted_retained_offset(uint64_t image_bytes)
{
   return (image_bytes + 7u) & ~UINT64_C(7);
}

static inline uint64_t
libagx_gen1_as_compacted_span(const struct libagx_gen1_as_metadata *metadata)
{
   return metadata->retained_bytes
      ? libagx_gen1_as_compacted_retained_offset(metadata->image_bytes) + metadata->retained_bytes
      : metadata->image_bytes;
}
#endif
#endif
