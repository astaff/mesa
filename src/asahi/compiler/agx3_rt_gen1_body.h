#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define AGX3_RT_GEN1_HIT_COMMIT_MAX_SIZE 256u
bool agx3_rt_gen1_hit_commit_generate(
   uint8_t out[AGX3_RT_GEN1_HIT_COMMIT_MAX_SIZE], size_t *length);
