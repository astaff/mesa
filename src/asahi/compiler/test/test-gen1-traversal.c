#include "agx3_rt_gen1_traversal.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>
int main(void) {
 uint8_t out[16],n;
 const uint8_t expected[]={0x5f,0x10,0x54,0,0,1,0x26,0,0xc7,0x4f,0x4c,0};
 struct agx3_rt_gen1_intersect_desc d={.bvh=96,.request=112,.context_uniform=126,.bvh_uniform=true,.request_uniform=true};
 assert(agx3_rt_gen1_encode_intersect(AGX_ISA_ENCODING_AGX3B,&d,out,&n));assert(n==sizeof(expected)&&!memcmp(out,expected,n));
 assert(!agx3_rt_gen1_encode_intersect(AGX_ISA_ENCODING_AGX3C,&d,out,&n));
 /* Fresh guarded candidate_ids G17P witness: register request and16-bit
  * context uniform123. The latter is valid at an odd half index. */
 const uint8_t candidate[]={0x5f,0x10,0x54,0,0,1,0x26,0x20,0x63,0x4f,0xcc,0};
 d=(struct agx3_rt_gen1_intersect_desc){.bvh=96,.request=50,.context_uniform=123,.bvh_uniform=true};
 assert(agx3_rt_gen1_encode_intersect(AGX_ISA_ENCODING_AGX3B,&d,out,&n));
 assert(n==sizeof(candidate)&&!memcmp(out,candidate,n));
 d.context_uniform=512;assert(!agx3_rt_gen1_encode_intersect(AGX_ISA_ENCODING_AGX3B,&d,out,&n));

 const uint8_t resume[]={0x5f,0x11,0x54,4,0,0};
 assert(agx3_rt_gen1_encode_continue(AGX_ISA_ENCODING_AGX3B,out,&n));assert(n==sizeof(resume)&&!memcmp(out,resume,n));
 const uint8_t first[]={0x0b,0x19,0x27,0,0x42,0,0,0,0x10,0};
 const uint8_t second[]={0x1b,0x1b,0x27,0,0x42,0,0,0,0x10,0};
 assert(agx3_rt_gen1_encode_launch_word(0,24,out,&n));assert(n==sizeof(first)&&!memcmp(out,first,n));
 assert(agx3_rt_gen1_encode_launch_word(1,26,out,&n));assert(n==sizeof(second)&&!memcmp(out,second,n));
 assert(!agx3_rt_gen1_encode_launch_word(2,0,out,&n));
 /* SR59 has already been zero-extended: publish its 32-bit register,
  * barrier, then consume staging4. Keep this sequence byte-exact. */
 uint8_t sequence[32];
 const uint8_t published[]={0x2b,0x14,0x21,0x04,
                           0x07,0x04,0x54,0x08,0x09,0x00,
                           0x5f,0x11,0x54,0x04,0x00,0x00};
 assert(agx3_rt_gen1_encode_resume(AGX_ISA_ENCODING_AGX3B,20,sequence,&n));
 assert(n==sizeof(published)&&!memcmp(sequence,published,n));
 assert(!agx3_rt_gen1_encode_resume(AGX_ISA_ENCODING_AGX3C,20,sequence,&n));
 assert(!agx3_rt_gen1_encode_resume(AGX_ISA_ENCODING_AGX3B,21,sequence,&n));
 assert(!agx3_rt_gen1_encode_resume(AGX_ISA_ENCODING_AGX3B,256,sequence,&n));
 const uint8_t refill[]={0x0b,0x14,0x21,0x04,
                         0x07,0x04,0x54,0x48,0x09,0x00,
                         0x5f,0x11,0x54,0x00,0x00,0x00};
 assert(agx3_rt_gen1_encode_refill(AGX_ISA_ENCODING_AGX3B,20,sequence,&n));
 assert(n==sizeof(refill)&&!memcmp(sequence,refill,n));
 assert(!agx3_rt_gen1_encode_refill(AGX_ISA_ENCODING_AGX3C,20,sequence,&n));
 puts("Gen1 intersect and resident-ID/barrier/continue sequence match native query");
}
