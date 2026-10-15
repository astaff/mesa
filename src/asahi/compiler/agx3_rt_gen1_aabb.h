#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "agx3_rt_gen1_entry.h"
#define AGX3_RT_GEN1_AABB_SLOT 109u
bool agx3_rt_gen1_aabb_body_generate(uint8_t *out, size_t capacity, size_t *length);
void agx3_rt_gen1_aabb_constants(const uint8_t **data, size_t *length);
bool agx3_rt_gen1_aabb_entry_generate(uint64_t slot_body_va, uint8_t out[AGX3_RT_GEN1_ENTRY_MAX_SIZE], size_t *length);
