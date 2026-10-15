#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define AGX3_RT_GEN1_INSTANCE_PUSH_SLOT 189u
bool agx3_rt_gen1_instance_push_body_generate(uint8_t *out, size_t capacity, size_t *length);
void agx3_rt_gen1_instance_push_constants(const uint8_t **data, size_t *length);
