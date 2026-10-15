#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
bool agx3_rt_gen1_clique_slot(unsigned index, unsigned *slot);
bool agx3_rt_gen1_clique_body_generate(unsigned slot, uint8_t *out, size_t capacity, size_t *length);
