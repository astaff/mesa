#include "agx3_rt_gen1_body.h"
#include "agx3_binder.h"
#include <string.h>
#define KV(f, v) {(f), (v)}
bool agx3_rt_gen1_hit_commit_generate(uint8_t out[AGX3_RT_GEN1_HIT_COMMIT_MAX_SIZE], size_t *length) {
   if (!out || !length) return false;
   uint8_t staged[AGX3_RT_GEN1_HIT_COMMIT_MAX_SIZE];size_t used=0;
#define OP(name, ep, ...) do { \
   const struct agx3_field_kv fields[] = {__VA_ARGS__}; uint8_t n; \
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B,name,ep,fields, \
      sizeof(fields)/sizeof(fields[0]),NULL,0,staged+used,&n)) return false; \
   used+=n; \
} while(0)
   OP("ldscs", 0, KV(9, 2), KV(10, 2), KV(12, 6), KV(23, 1), KV(24, 1), KV(38, 1), KV(39, 1), KV(54, 1), KV(68, 1), KV(71, 24588));
   OP("rld", 0, KV(2, 2), KV(9, 2), KV(10, 1), KV(12, 1), KV(21, 58), KV(23, 7), KV(24, 1), KV(38, 1));
   OP("bit_and", 1, KV(2, 6), KV(9, 2), KV(10, 1), KV(11, 1), KV(16, 2), KV(23, 2), KV(24, 1), KV(27, 1), KV(31, 6), KV(37, 3), KV(38, 1), KV(44, 1));
   OP("bit_and", 1, KV(2, 2), KV(9, 2), KV(10, 2), KV(16, 6), KV(23, 2), KV(24, 1), KV(31, 5), KV(37, 3), KV(38, 1));
   OP("imadd", 0, KV(2, 10), KV(9, 2), KV(10, 2), KV(16, 1), KV(23, 2), KV(24, 1), KV(27, 1), KV(31, 2), KV(38, 2), KV(39, 2), KV(42, 1), KV(53, 3), KV(54, 2), KV(65, 1));
   OP("bit_and", 1, KV(2, 12), KV(9, 2), KV(10, 2), KV(16, 6), KV(23, 2), KV(24, 1), KV(29, 31), KV(37, 1), KV(38, 2));
   OP("ldscs", 0, KV(2, 2), KV(9, 2), KV(10, 2), KV(12, 6), KV(16, 12), KV(23, 2), KV(24, 2), KV(31, 10), KV(38, 2), KV(39, 2), KV(54, 1), KV(68, 1), KV(70, 3), KV(71, 260));
   OP("rld", 0, KV(2, 4), KV(9, 2), KV(10, 1), KV(12, 1), KV(21, 57), KV(23, 7), KV(24, 1), KV(38, 1));
   OP("rld", 0, KV(2, 8), KV(9, 2), KV(10, 1), KV(12, 2), KV(21, 7), KV(23, 7), KV(24, 1), KV(38, 1));
   OP("ldimm", 1, KV(9, 2), KV(10, 2), KV(14, 1073741824));
   OP("bit_or", 1, KV(2, 2), KV(9, 2), KV(10, 2), KV(16, 2), KV(23, 2), KV(24, 2), KV(27, 1), KV(31, 8), KV(37, 3), KV(38, 2), KV(49, 1));
   OP("imadd", 0, KV(2, 4), KV(9, 2), KV(10, 2), KV(16, 4), KV(23, 2), KV(24, 1), KV(27, 1), KV(32, 4), KV(38, 3), KV(39, 1), KV(53, 1), KV(54, 1), KV(60, 1));
   OP("clr", 0, KV(2, 9), KV(9, 2), KV(10, 1));
   OP("iadd", 0, KV(2, 6), KV(9, 2), KV(10, 1), KV(16, 6), KV(23, 2), KV(24, 1), KV(27, 1), KV(32, 7), KV(38, 3), KV(39, 1));
   OP("clr", 0, KV(2, 7), KV(9, 2), KV(10, 1));
   OP("stscs", 0, KV(9, 2), KV(10, 2), KV(16, 12), KV(23, 2), KV(24, 2), KV(27, 1), KV(31, 10), KV(38, 2), KV(39, 2), KV(42, 1), KV(54, 1), KV(68, 3), KV(70, 3), KV(71, 256));
   OP("stscs", 0, KV(2, 6), KV(9, 2), KV(10, 2), KV(16, 8), KV(23, 2), KV(24, 2), KV(27, 1), KV(31, 4), KV(38, 2), KV(39, 2), KV(42, 1), KV(54, 1), KV(61, 1), KV(68, 1), KV(70, 2));
   OP("rld", 0, KV(2, 4), KV(9, 2), KV(10, 1), KV(12, 1), KV(21, 57), KV(23, 7), KV(24, 1), KV(38, 1));
   OP("ldimm", 1, KV(9, 2), KV(10, 2), KV(11, 1), KV(14, 192));
   OP("ldimm", 1, KV(2, 2), KV(9, 2), KV(10, 2), KV(11, 1), KV(14, 512));
   OP("imadd", 0, KV(9, 2), KV(10, 2), KV(16, 4), KV(23, 2), KV(24, 1), KV(27, 1), KV(31, 2), KV(38, 2), KV(39, 2), KV(42, 1), KV(53, 2), KV(54, 2), KV(57, 1), KV(60, 1));
   OP("ldscs", 0, KV(10, 2), KV(23, 1), KV(24, 1), KV(28, 1), KV(38, 2), KV(39, 2), KV(42, 1), KV(54, 1), KV(68, 3), KV(80, 1));
   OP("barrier", 0, KV(25, 1), KV(30, 1), KV(39, 1));
   {uint8_t n;if(!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B,"end",0,NULL,0,NULL,0,staged+used,&n))return false;used+=n;}
#undef OP
   if (used > sizeof(staged)) return false;
   memcpy(out,staged,used);*length=used;return true;
}
