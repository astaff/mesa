#include "agx3_rt_gen1_instance_push.h"
#include "agx3_binder.h"
#include "agx3_cf_core.h"
#include <string.h>
#define CAPACITY 16384u
#include "agx3_rt_gen1_instance_push_ops.h"
#define PUSH_COUNT (sizeof(agx3_rt_gen1_instance_push_ops)/sizeof(agx3_rt_gen1_instance_push_ops[0]))
struct stream {uint8_t bytes[CAPACITY];size_t size;bool ok;};
static void append(struct stream *s,const uint8_t *bytes,unsigned n){if(!n||n>16||s->size+n>CAPACITY){s->ok=false;return;}memcpy(s->bytes+s->size,bytes,n);s->size+=n;}
#define CHECK(x) do { if (!(x)) { s->ok = false; return; } } while (0)
static void body(struct stream *s){
 size_t positions[PUSH_COUNT];
 for(unsigned j=0;j<PUSH_COUNT;j++){
  const struct agx3_rt_gen1_native_op *op=&agx3_rt_gen1_instance_push_ops[j];positions[j]=s->size;
  uint8_t bytes[16],n;
  if(op->target!=UINT32_MAX){CHECK(agx3_cf_branch(AGX_ISA_ENCODING_AGX3B,true,bytes,&n));}
  else{CHECK(agx3_bind_compose(AGX_ISA_ENCODING_AGX3B,op->name,op->ep,op->fields,op->nfields,NULL,0,bytes,&n));}
  CHECK(n==op->size);append(s,bytes,n);
 }
 for(unsigned j=0;j<PUSH_COUNT;j++)if(agx3_rt_gen1_instance_push_ops[j].target!=UINT32_MAX){size_t at=positions[j],target=positions[agx3_rt_gen1_instance_push_ops[j].target];CHECK(agx3_cf_patch_branch(AGX_ISA_ENCODING_AGX3B,true,s->bytes+at,(int64_t)target-(int64_t)at));}
}
#undef CHECK
bool
agx3_rt_gen1_instance_push_body_generate(uint8_t *out, size_t capacity, size_t *length)
{
   static const uint8_t prefix[4] = {0x0e, 0, 0, 0};
   struct stream s = {.ok = true};
   body(&s);
   if (!s.ok || 64 + s.size > capacity) return false;
   memset(out, 0, 64);
   memcpy(out, prefix, sizeof(prefix));
   for (unsigned i = 2; i < 32; ++i) out[2 * i] = 0x06;
   memcpy(out + 64, s.bytes, s.size);
   *length = 64 + s.size;
   return true;
}
static const uint8_t instance_push_constants[112] = {0xfe,0xff,0x7f,0x3f,0x01,0x00,0x80,0x3f,0x00,0x80,0x00,0x00,0x00,0x02,0x00,0x20,0x00,0x10,0x00,0x00,0xe0,0x0f,0xff,0x0f,0x00,0x5c,0x00,0x7c,0x00,0xc0,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x01,0x00,0x01,0x00,0x02,0x00,0x01,0x00,0xff,0xff,0x01,0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x08,0x00,0x00,0x20,0x08,0x00,0x00,0x00,0x10,0x00,0x00,0x00,0x3f,0x00,0x00,0x00,0x40,0x00,0xff,0xff,0xff,0x00,0x00,0x00,0x00,0x20,0x00,0x00,0xa0,0x34,0x00,0x00,0x7f,0x43,0xff,0xff,0x07,0xff,0x00,0x00,0x0f,0xff,0xff,0xff,0xf3,0xff,0xff,0xff,0xfd,0xff,0xff,0x3f,0xff,0xff};
void
agx3_rt_gen1_instance_push_constants(const uint8_t **data, size_t *length)
{
   *data = instance_push_constants;
   *length = sizeof(instance_push_constants);
}
