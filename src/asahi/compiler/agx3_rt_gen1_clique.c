#include "agx3_rt_gen1_clique.h"
#include "agx3_binder.h"
#include "agx3_cf_core.h"
#include <string.h>
#define CAPACITY 16384u
#include "agx3_rt_gen1_clique_ops.h"
#define PROGRAM_COUNT (sizeof(agx3_rt_gen1_clique_programs)/sizeof(agx3_rt_gen1_clique_programs[0]))
struct stream {uint8_t bytes[CAPACITY];size_t size;bool ok;};
static void append(struct stream *s,const uint8_t *bytes,unsigned n){if(!n||n>16||s->size+n>CAPACITY){s->ok=false;return;}memcpy(s->bytes+s->size,bytes,n);s->size+=n;}
#define CHECK(x) do { if (!(x)) { s->ok = false; return; } } while (0)
static void body(struct stream *s,const struct agx3_rt_gen1_native_op *ops,unsigned count){
 size_t positions[1024];
 if(count>1024){s->ok=false;return;}
 for(unsigned j=0;j<count;j++){
  const struct agx3_rt_gen1_native_op *op=&ops[j];positions[j]=s->size;
  uint8_t bytes[16],n;
  if(op->target!=UINT32_MAX){CHECK(agx3_cf_branch(AGX_ISA_ENCODING_AGX3B,true,bytes,&n));}
  else{CHECK(agx3_bind_compose(AGX_ISA_ENCODING_AGX3B,op->name,op->ep,op->fields,op->nfields,NULL,0,bytes,&n));}
  CHECK(n==op->size);append(s,bytes,n);
 }
 for(unsigned j=0;j<count;j++)if(ops[j].target!=UINT32_MAX){size_t at=positions[j],target=positions[ops[j].target];CHECK(agx3_cf_patch_branch(AGX_ISA_ENCODING_AGX3B,true,s->bytes+at,(int64_t)target-(int64_t)at));}
}
#undef CHECK
bool
agx3_rt_gen1_clique_slot(unsigned index, unsigned *slot)
{
   if (index >= PROGRAM_COUNT || !slot) return false;
   *slot = agx3_rt_gen1_clique_programs[index].slot;
   return true;
}
bool
agx3_rt_gen1_clique_body_generate(unsigned slot, uint8_t *out, size_t capacity, size_t *length)
{
   static const uint8_t prefix[4] = {0x0e, 0, 0, 0};
   for (unsigned i = 0; i < PROGRAM_COUNT; ++i) {
      if (agx3_rt_gen1_clique_programs[i].slot != slot) continue;
      struct stream s = {.ok = true};
      body(&s, agx3_rt_gen1_clique_programs[i].ops, agx3_rt_gen1_clique_programs[i].count);
      if (!s.ok || 64 + s.size > capacity) return false;
      memset(out, 0, 64);
      memcpy(out, prefix, sizeof(prefix));
      for (unsigned k = 2; k < 32; ++k) out[2 * k] = 0x06;
      memcpy(out + 64, s.bytes, s.size);
      *length = 64 + s.size;
      return true;
   }
   return false;
}
