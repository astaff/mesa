#include "agx3_scs_core.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
int main(void) {
 const uint8_t native[]={0x5f,0x02,0x04,0x00,0x02,0x00,0x00,0x00,0x80,0x8b,0x03,0x00,0x20,0x00};
 struct agx3_scs_desc d;assert(agx3_scs_decode_load(AGX_ISA_ENCODING_AGX3B,native,sizeof(native),&d));assert(d.b_status_load&&d.width==AGX3_MEM_W8&&d.base_register==2&&d.byte_offset==7);
 uint8_t out[AGX3_SCS_MAX_LENGTH],n;assert(agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3B,&d,out,&n));assert(n==sizeof(native)&&!memcmp(native,out,n));
 puts("Gen1 special byte-status load matches native");return 0;
}
