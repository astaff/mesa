#pragma once
#include "asahi/lib/agx_chip.h"
#include <stdbool.h>
#include <stdint.h>
struct agx3_rt_gen1_intersect_desc {
   uint16_t bvh, request, context_uniform;
   bool bvh_uniform, request_uniform;
};
bool agx3_rt_gen1_encode_intersect(enum agx_isa_encoding encoding,
   const struct agx3_rt_gen1_intersect_desc *desc, uint8_t out[16], uint8_t *length);
bool agx3_rt_gen1_encode_continue(enum agx_isa_encoding encoding,
   uint8_t out[16], uint8_t *length);
/* Fused publication, SCS barrier and continuation. Source is a 32-bit ID. */
bool agx3_rt_gen1_encode_resume(enum agx_isa_encoding encoding,
   uint16_t resident_source, uint8_t out[32], uint8_t *length);
bool agx3_rt_gen1_encode_launch_word(unsigned word, uint16_t source,
   uint8_t out[16], uint8_t *length);

/* Publish resident ID to staging0 and issue the queue-refill continuation. */
bool agx3_rt_gen1_encode_refill(enum agx_isa_encoding encoding,
   uint16_t resident_source, uint8_t out[32], uint8_t *length);
