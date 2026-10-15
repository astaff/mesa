/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "asahi/lib/agx_chip.h"
#ifdef __cplusplus
extern "C" {
#endif



bool agx3_mesh_allocate_uvb(enum agx_isa_encoding enc, uint16_t operand0,
                            uint16_t operand1, bool first_last_use, uint8_t out[8], uint8_t *length);
bool agx3_mesh_uvst(enum agx_isa_encoding enc, uint16_t descriptor,
                   bool primitive, bool last_use, uint8_t out[8], uint8_t *length);
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif




bool agx3_mesh_object_childcount(enum agx_isa_encoding enc, uint8_t out[8],
                                  uint8_t *length);
bool agx3_mesh_passthrough_object(uint16_t grid_uniform, uint8_t out[64],
                                 uint8_t *length);
#ifdef __cplusplus
}
#endif
