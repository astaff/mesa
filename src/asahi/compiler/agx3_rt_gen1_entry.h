#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define AGX3_RT_GEN1_ENTRY_MAX_SIZE 512u
#define AGX3_RT_GEN1_ENTRY_MAX_UNIFORMS 36u
struct agx3_rt_gen1_uniform {
   uint16_t uniform_half;
   uint16_t register_half;
};
struct agx3_rt_gen1_entry_desc {
   uint64_t constants_va;
   uint64_t shader_va;
   uint64_t branch_va;
   const struct agx3_rt_gen1_uniform *uniforms;
   uint8_t constant_dwords;
   uint8_t fetch_span;
   uint8_t uniform_count;
   bool context48;
};
bool agx3_rt_gen1_entry_generate(const struct agx3_rt_gen1_entry_desc *desc,
                                uint8_t out[AGX3_RT_GEN1_ENTRY_MAX_SIZE],
                                size_t *length);
