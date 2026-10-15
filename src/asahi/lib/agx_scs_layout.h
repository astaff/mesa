/* SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/* M5's ScsLayoutGen2, as consumed by Apple's per-kick SCS parameters.
 * Numeric section identifiers follow getSectionEnd; 7..9 are empty sections.
 * This describes allocation, not the shader's compiler-chosen payload map.
 */
struct agx_scs_layout_gen2 {
   uint64_t packed;
   uint32_t end[13];
};

/* G17P's Gen1 uses the same packed fields and section numbering. Section 4
 * is 64 KiB rather than 128 KiB, and section 6 has no per-level storage. */
bool agx_scs_layout_gen1_init(uint64_t packed,
                            struct agx_scs_layout_gen2 *layout);
void agx_scs_layout_gen1_parameters(const struct agx_scs_layout_gen2 *layout,
                                   uint8_t parameters[48]);

/* Returns false if the native 32-bit section offsets would overflow. */
bool agx_scs_layout_gen2_init(uint64_t packed,
                            struct agx_scs_layout_gen2 *layout);

/* Write the complete 48-byte, initially-zero per-kick parameter record.
 * The first eight bytes are reserved for runtime state, initially zero.
 */
void agx_scs_layout_gen2_parameters(const struct agx_scs_layout_gen2 *layout,
                                  uint8_t parameters[48]);
