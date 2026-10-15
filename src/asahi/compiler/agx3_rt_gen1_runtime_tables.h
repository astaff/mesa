#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "agx3_rt_gen1_entry.h"
#define AGX3_RT_GEN1_RUNTIME_USER_IMAGE_BYTES 0x60000u
#define AGX3_RT_GEN1_RUNTIME_UNIFORM_BYTES 320u
bool agx3_rt_gen1_runtime_entry_generate(unsigned slot, uint8_t out[AGX3_RT_GEN1_ENTRY_MAX_SIZE], size_t *length);
bool agx3_rt_gen1_runtime_user_image(uint64_t packed_layout, uint8_t *out, size_t capacity);
void agx3_rt_gen1_runtime_uniforms(uint8_t out[AGX3_RT_GEN1_RUNTIME_UNIFORM_BYTES]);
bool agx3_rt_gen1_runtime_constants(unsigned slot, const uint8_t **data, size_t *length);
