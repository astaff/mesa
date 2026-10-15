#include "agx3_rt_gen1_triangle.h"
#include "agx3_binder.h"
#include "agx3_alu_core.h"
#include "agx3_isel_core.h"
#include "agx3_cf_core.h"
#include "agx3_pack.h"
#include "agx3_rt_gen1_entry.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define CAPACITY 16384u
#include "agx3_rt_gen1_triangle_ops.h"
#define TRI_COUNT (sizeof(agx3_rt_gen1_triangle_ops)/sizeof(agx3_rt_gen1_triangle_ops[0]))
struct stream {uint8_t bytes[CAPACITY];size_t size;bool ok;};
static void append(struct stream *s,const uint8_t *bytes,unsigned n,const char *name,int original){(void)name;(void)original;if(!n||n>16||s->size+n>CAPACITY){s->ok=false;return;}memcpy(s->bytes+s->size,bytes,n);s->size+=n;}
#undef assert
#define assert(x) do { if (!(x)) { s->ok = false; return; } } while (0)
static void fsel(struct stream *s,unsigned dest,unsigned a,unsigned b,unsigned cond,unsigned t,bool ti,unsigned f,bool fi,bool absolute,const char *name){
 struct agx3_fsel_desc d={.dest=dest,.dest_16=true,.dest_cache=true,.cmp0=a,.cmp1=b,.cmp0_abs=absolute,.sel0_kind=ti?AGX3_FSEL_SEL_IMMEDIATE:AGX3_FSEL_SEL_REGISTER,.sel1_kind=fi?AGX3_FSEL_SEL_IMMEDIATE:AGX3_FSEL_SEL_REGISTER,.sel0=t,.sel1=f,.sel_16=true,.condition=cond};
 uint8_t bytes[AGX3_ALU_MAX_LENGTH],n;assert(agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B,&d,bytes,&n));append(s,bytes,n,name,-1);
}
static void isel(struct stream *s,unsigned dest,unsigned a,unsigned b,bool narrow,unsigned t,bool ti,unsigned f,bool fi,const char *name){
 struct agx3_isel_desc d={.dest=dest,.dest_16=true,.cmp0=a,.cmp1=b,.cond=AGX3_ISEL_ICOND_EQ,.cmp1_immediate=true,.cmp_16=narrow,.explicit_hints=true,.dest_cache=true,.true_kind=ti?AGX3_ISEL_IMMEDIATE:AGX3_ISEL_REGISTER,.false_kind=fi?AGX3_ISEL_IMMEDIATE:AGX3_ISEL_REGISTER,.sel_true=t,.sel_false=f};
 uint8_t bytes[AGX3_ISEL_MAX_LENGTH],n;assert(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B,&d,bytes,&n));append(s,bytes,n,name,-1);
}
static void imm16(struct stream *s,unsigned dest,unsigned value,const char *name){
 const struct agx3_field_kv fields[]={{2,dest},{9,2},{10,1},{14,value}};
 uint8_t bytes[16],n;assert(agx3_bind_compose(AGX_ISA_ENCODING_AGX3B,"ldimm",1,fields,4,NULL,0,bytes,&n));append(s,bytes,n,name,-1);
}
static void imm32(struct stream *s,unsigned dest,uint32_t value,const char *name){struct agx3_encoded_ldimm i;assert(agx3_pack_ldimm_u32(AGX_ISA_ENCODING_AGX3B,dest,value,&i));append(s,i.bytes,i.length,name,-1);}
static void bitreg(struct stream *s,enum agx3_alu_op op,unsigned dest,unsigned a,unsigned b,const char *name){const uint16_t src[]={a,b};const enum agx3_alu_hint hints[]={AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE};uint8_t bytes[16],n;assert(agx3_alu_bit_reg16(AGX_ISA_ENCODING_AGX3B,op,dest,src,hints,0,bytes,&n));append(s,bytes,n,name,-1);}
static void bitimm(struct stream *s,unsigned dest,unsigned a,unsigned mask,const char *name){uint8_t bytes[16],n;assert(mask<=255);assert(agx3_alu_bit_imm(AGX_ISA_ENCODING_AGX3B,AGX3_ALU_BAND,true,dest,a,mask,AGX3_ALU_HINT_NONE,0,bytes,&n));append(s,bytes,n,name,-1);}
static void setflag(struct stream *s,unsigned flag,const char *name){imm16(s,98,flag,name);isel(s,98,97,0,true,0,true,98,false,name);bitreg(s,AGX3_ALU_BOR,96,96,98,name);}
static void edge(struct stream *s,unsigned ax,unsigned ay,unsigned bx,unsigned by,unsigned flag){
 fsel(s,97,by,ay,1,1,true,0,true,false,"edge-dy-negative");
 fsel(s,97,bx,ax,0,97,false,0,true,false,"edge-vertical");
 fsel(s,97,bx,ax,1,1,true,97,false,false,"edge-dx-negative");setflag(s,flag,"edge-eligible");
}
static void vertex(struct stream *s,unsigned ax,unsigned bx,unsigned cx,unsigned flag){fsel(s,97,bx,ax,6,1,true,0,true,false,"vertex-left-inclusive");fsel(s,97,cx,ax,2,97,false,0,true,false,"vertex-right-exclusive");setflag(s,flag,"vertex-eligible");}
static void zero_factor(struct stream *s,unsigned a,unsigned b,unsigned flag){
 uint8_t bytes[16],n;
 assert(agx3_alu_shift_imm32(AGX_ISA_ENCODING_AGX3B,true,98,a,1,AGX3_ALU_HINT_NONE,true,0,bytes,&n));append(s,bytes,n,"factor-a-bitwise-abs-zero",-1);
 isel(s,97,98,0,false,1,true,0,true,"factor-a-zero");
 assert(agx3_alu_shift_imm32(AGX_ISA_ENCODING_AGX3B,true,98,b,1,AGX3_ALU_HINT_NONE,true,0,bytes,&n));append(s,bytes,n,"factor-b-bitwise-abs-zero",-1);
 isel(s,97,98,0,false,1,true,97,false,"factor-b-zero");setflag(s,flag,"zero-factor-proof");
}
static void subnormal_coordinate(struct stream *s,unsigned coordinate){
 uint8_t bytes[16],n;assert(agx3_alu_shift_imm32(AGX_ISA_ENCODING_AGX3B,true,98,coordinate,1,AGX3_ALU_HINT_NONE,true,0,bytes,&n));append(s,bytes,n,"coordinate-absolute-bits",-1);
 isel(s,97,98,0,false,0,true,1,true,"coordinate-bitwise-nonzero");
 assert(agx3_alu_shift_imm32(AGX_ISA_ENCODING_AGX3B,false,98,98,24,AGX3_ALU_HINT_NONE,true,0,bytes,&n));append(s,bytes,n,"coordinate-exponent",-1);
 isel(s,97,98,0,false,97,false,0,true,"coordinate-is-subnormal");setflag(s,32768,"preserve-native-subnormal-coordinate");
}
static void prepare_policy(struct stream *s){
 imm16(s,96,0,"ownership-state-init");
 subnormal_coordinate(s,0);subnormal_coordinate(s,18);subnormal_coordinate(s,14);subnormal_coordinate(s,2);subnormal_coordinate(s,4);subnormal_coordinate(s,22);
 edge(s,14,2,4,22,8);edge(s,4,22,0,18,16);edge(s,0,18,14,2,32);
 vertex(s,0,14,4,64);vertex(s,14,4,0,128);vertex(s,4,0,14,256);
 zero_factor(s,2,4,512);zero_factor(s,22,14,1024);zero_factor(s,22,0,2048);zero_factor(s,18,4,4096);zero_factor(s,18,14,8192);zero_factor(s,2,0,16384);
}
static void finite_product(struct stream *s,unsigned product){
 uint8_t bytes[16],n;assert(agx3_alu_shift_imm32(AGX_ISA_ENCODING_AGX3B,false,98,product,23,AGX3_ALU_HINT_NONE,true,0,bytes,&n));append(s,bytes,n,"product-exponent",-1);
 bitimm(s,98,98,255,"product-exponent-mask");isel(s,97,98,255,true,0,true,97,false,"finite-product-range");
}
static void residual_zero(struct stream *s,unsigned p0,unsigned p1,unsigned q0,unsigned q1,unsigned flag,unsigned proof0,unsigned proof1){
 fsel(s,97,p0,p1,0,1,true,0,true,false,"rounded-products-equal");
 fsel(s,97,q0,q1,0,97,false,0,true,false,"product-residuals-equal");setflag(s,flag,"exact-zero-edge");
 imm32(s,98,UINT32_C(0x18800000),"residual-safe-product-threshold");
 fsel(s,97,p0,98,5,1,true,0,true,true,"product-a-normal-residual-range");
 fsel(s,99,p1,98,5,1,true,0,true,true,"product-b-normal-residual-range");
 imm16(s,98,proof0,"factor-a-proof-mask");bitreg(s,AGX3_ALU_BAND,98,96,98,"factor-a-proof");isel(s,97,98,0,true,97,false,1,true,"product-a-safe-or-zero-factor");
 imm16(s,98,proof1,"factor-b-proof-mask");bitreg(s,AGX3_ALU_BAND,98,96,98,"factor-b-proof");isel(s,99,98,0,true,99,false,1,true,"product-b-safe-or-zero-factor");
 bitreg(s,AGX3_ALU_BAND,97,97,99,"both-products-qualified");finite_product(s,p0);finite_product(s,p1);
 imm16(s,98,flag,"exact-edge-mask");bitreg(s,AGX3_ALU_BAND,98,96,98,"exact-edge-test");isel(s,97,97,0,true,98,false,0,true,"unsafe-exact-edge");setflag(s,32768,"preserve-native-unsafe-edge");
}
static void finish_policy(struct stream *s){
 isel(s,99,72,0,true,0,true,1,true,"save-native-valid");
 bitimm(s,97,96,9,"u-zero-and-eligible");isel(s,97,97,9,true,1,true,0,true,"u-owned");
 bitimm(s,98,96,18,"v-zero-and-eligible");isel(s,97,98,18,true,1,true,97,false,"v-owned-or-u");
 bitimm(s,98,96,36,"w-zero-and-eligible");isel(s,97,98,36,true,1,true,97,false,"w-owned-or-uv");
 bitimm(s,98,96,7,"zero-edge-mask");isel(s,97,98,0,true,1,true,97,false,"no-zero-edge-preserves-native");bitreg(s,AGX3_ALU_BAND,72,72,97,"edge-owner-valid");
 bitimm(s,97,96,7,"vertex-zero-mask");
 bitimm(s,98,96,64,"vertex-a-owner");isel(s,98,97,6,true,98,false,1,true,"vertex-a-only-on-vw-zero");isel(s,72,98,0,true,0,true,72,false,"vertex-a-valid");
 bitimm(s,98,96,128,"vertex-b-owner");isel(s,98,97,5,true,98,false,1,true,"vertex-b-only-on-uw-zero");isel(s,72,98,0,true,0,true,72,false,"vertex-b-valid");
 imm16(s,98,256,"vertex-c-mask");bitreg(s,AGX3_ALU_BAND,98,96,98,"vertex-c-owner");isel(s,98,97,3,true,98,false,1,true,"vertex-c-only-on-uv-zero");isel(s,72,98,0,true,0,true,72,false,"vertex-c-valid");
 imm16(s,98,32768,"unsafe-edge-mask");bitreg(s,AGX3_ALU_BAND,98,96,98,"unsafe-edge-test");isel(s,72,98,0,true,72,false,99,false,"qualified-policy-or-native");isel(s,72,97,7,true,99,false,72,false,"degenerate-projection-preserves-native");
}
static void body(struct stream *s,bool corrected){
 size_t positions[TRI_COUNT];
 for(unsigned j=0;j<TRI_COUNT;j++){
  const struct agx3_rt_gen1_native_op *op=&agx3_rt_gen1_triangle_ops[j];positions[j]=s->size;
  if(corrected&&op->offset==620)prepare_policy(s);
  uint8_t bytes[16],n;
  if(op->target!=UINT32_MAX)assert(agx3_cf_branch(AGX_ISA_ENCODING_AGX3B,true,bytes,&n));
  else assert(agx3_bind_compose(AGX_ISA_ENCODING_AGX3B,op->name,op->ep,op->fields,op->nfields,NULL,0,bytes,&n));
  assert(n==op->size);append(s,bytes,n,op->name,op->offset);
  if(corrected){switch(op->offset){case 648:residual_zero(s,72,10,76,78,1,512,1024);break;case 700:residual_zero(s,80,82,84,86,2,2048,4096);break;case 722:residual_zero(s,88,18,90,92,4,8192,16384);break;case 908:finish_policy(s);break;default:break;}}
 }
 for(unsigned j=0;j<TRI_COUNT;j++)if(agx3_rt_gen1_triangle_ops[j].target!=UINT32_MAX){size_t at=positions[j],target=positions[agx3_rt_gen1_triangle_ops[j].target];assert(agx3_cf_patch_branch(AGX_ISA_ENCODING_AGX3B,true,s->bytes+at,(int64_t)target-(int64_t)at));}
}
#undef assert
bool
agx3_rt_gen1_triangle_body_generate(uint8_t *out, size_t capacity, size_t *length)
{
   static const uint8_t prefix[4] = {0x0e, 0, 0, 0};
   struct stream s = {.ok = true};
   body(&s, true);
   if (!s.ok || 64 + s.size > capacity) return false;
   memset(out, 0, 64);
   memcpy(out, prefix, sizeof(prefix));
   for (unsigned i = 2; i < 32; ++i) out[2 * i] = 0x06;
   memcpy(out + 64, s.bytes, s.size);
   *length = 64 + s.size;
   return true;
}
static const uint8_t triangle_constants[92] = {0x00,0x00,0x00,0x40,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x3f,0x00,0x01,0x00,0x02,0x00,0x00,0x40,0x00,0xff,0xff,0xff,0xbf,0x01,0x08,0x01,0x10,0x00,0x20,0x00,0x80,0x00,0x00,0xe0,0x01,0x00,0x00,0xc0,0x7f,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0xc0,0xff,0xff,0xff,0xe0,0xe0,0x0f,0xff,0x0f,0x00,0x10,0x01,0x18,0x01,0x20,0x00,0x28,0x01,0x28,0x00,0x40,0x00,0xc0,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x20,0x00,0x01,0x00,0x80,0x7f,0xff,0xff,0xff,0x7f,0xff,0xff,0x1f,0xfe};
void
agx3_rt_gen1_triangle_constants(const uint8_t **data, size_t *length)
{
   *data = triangle_constants;
   *length = sizeof(triangle_constants);
}
bool
agx3_rt_gen1_triangle_entry_generate(uint64_t slot_body_va, uint8_t out[AGX3_RT_GEN1_ENTRY_MAX_SIZE], size_t *length)
{
   static const struct agx3_rt_gen1_uniform uniforms[] = {{0,20},{2,22},{52,72},{54,74},{44,64},{46,66},{48,68},{50,70},{36,56},{38,58},{40,60},{42,62},{28,48},{30,50},{32,52},{34,54},{20,40},{22,42},{24,44},{26,46},{12,32},{14,34},{16,36},{18,38},{4,24},{6,26},{8,28},{10,30}};
   const struct agx3_rt_gen1_entry_desc d = {.constants_va = slot_body_va + 0x10000, .shader_va = slot_body_va + 64, .branch_va = slot_body_va, .uniforms = uniforms, .constant_dwords = 26, .fetch_span = 3, .uniform_count = 28, .context48 = false};
   return agx3_rt_gen1_entry_generate(&d, out, length);
}
