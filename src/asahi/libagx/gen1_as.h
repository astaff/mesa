/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef LIBAGX_GEN1_AS_H
#define LIBAGX_GEN1_AS_H

#include <stdint.h>
#include "gen1_as_metadata.h"

/* agx3_envelope.comp publishes these offsets relative to the native root.
 * Gen1 BVHState masks fourteen address bits, so a Vulkan AS allocation also
 * reserves the worst-case prefix needed to align its native root to 16 KiB.
 */
#define LIBAGX_GEN1_AS_ALIGNMENT 16384u
#define LIBAGX_GEN1_AS_ALIGNMENT_SLACK (LIBAGX_GEN1_AS_ALIGNMENT - 1u)
#define LIBAGX_GEN1_AS_EMPTY_BYTES 192u
#define LIBAGX_GEN1_AS_SIZE_WORD 9u
#define LIBAGX_GEN1_AS_BASE_WORD 18u
#define LIBAGX_GEN1_AS_ID_WORD 22u

#define LIBAGX_GEN1_AS_SERIAL_HEADER_BYTES 56u
#define LIBAGX_GEN1_AS_SERIAL_FORMAT_UUID_BYTES \
   {0x0e,0x8f,0x4b,0x53,0x91,0x1a,0x4c,0x50, \
    0x8a,0xb7,0x61,0x70,0x73,0x00,0x01,0x01}

struct libagx_gen1_as_address {
   uint64_t address;
   uint32_t resource_id;
   uint32_t reserved;
};

static inline uint64_t
libagx_gen1_as_serial_image_offset(uint64_t handles)
{
   return LIBAGX_GEN1_AS_SERIAL_HEADER_BYTES +
          LIBAGX_GEN1_AS_METADATA_STRIDE + 16u * handles;
}

static inline uint64_t
libagx_gen1_as_serial_size(const struct libagx_gen1_as_metadata *metadata,
                          uint64_t handles)
{
   return libagx_gen1_as_serial_image_offset(handles) +
          libagx_gen1_as_compacted_span(metadata);
}

static inline uint64_t
libagx_gen1_as_native_address(uint64_t address)
{
   return (address + LIBAGX_GEN1_AS_ALIGNMENT_SLACK) &
          ~((uint64_t)LIBAGX_GEN1_AS_ALIGNMENT_SLACK);
}

static inline uint64_t
libagx_gen1_as_image_bytes(uint64_t header_size)
{
   /* Empty-AS construction writes a zero size with NaN bounds. */
   return header_size ? header_size : LIBAGX_GEN1_AS_EMPTY_BYTES;
}

static inline uint64_t
libagx_gen1_as_property_size(uint64_t header_size)
{
   return libagx_gen1_as_image_bytes(header_size) +
          LIBAGX_GEN1_AS_ALIGNMENT_SLACK;
}

static inline uint64_t
libagx_gen1_as_relocate_word(uint64_t value, uint64_t word,
                           uint64_t header_size, uint64_t destination,
                           uint32_t destination_id)
{
   if (header_size && word == LIBAGX_GEN1_AS_BASE_WORD)
      return destination;
   if (header_size && word == LIBAGX_GEN1_AS_ID_WORD)
      return (value & UINT64_C(0xffffffff00000000)) | destination_id;
   return value;
}

#endif
