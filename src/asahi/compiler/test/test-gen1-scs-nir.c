#include "agx_compile.h"
#include "agx_debug.h"
#include "agx_nir.h"
#include "agx3_scs_core.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/glsl_types.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static bool check(unsigned mode,unsigned count,enum agx_isa_encoding encoding,int offset) {
 nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,&agx_nir_options,"Gen1 SCS NIR");
 b.shader->info.workgroup_size[0]=64;b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
 nir_def *id=nir_load_local_invocation_index(&b),*v[4];
 for(unsigned i=0;i<count;i++)v[i]=mode==0?nir_imm_int(&b,17*i+23):nir_iadd_imm(&b,id,17*i+23);
 nir_def *value=nir_vec(&b,v,count),*out;
 if(mode==0){nir_store_scs_agx(&b,value,.base=offset);out=nir_load_scs_agx(&b,count,32,.base=offset);}
 else if(mode==1){nir_def *base=nir_imul_imm(&b,id,16),*index=nir_iand_imm(&b,id,0);nir_store_scs_indexed_agx(&b,value,base,index,.base=offset,.component=3);out=nir_load_scs_indexed_agx(&b,count,32,base,index,.base=offset,.component=3);}
 else if(mode==2){nir_def *base=nir_imul_imm(&b,id,16);nir_store_scs_base_agx(&b,value,base,.base=offset,.format=PIPE_FORMAT_R32_UINT,.component=2);out=nir_load_scs_base_agx(&b,count,32,base,.base=offset,.format=PIPE_FORMAT_R32_UINT,.component=2);}
 else {nir_def *base=nir_load_preamble(&b,1,64,.base=122),*index=nir_imul_imm(&b,id,16);nir_store_scs_uniform_agx(&b,value,base,index,.base=offset,.format=PIPE_FORMAT_R32_UINT,.component=2);out=nir_load_scs_uniform_agx(&b,count,32,base,index,.base=offset,.format=PIPE_FORMAT_R32_UINT,.component=2);}
 nir_def *address=nir_iadd_imm(&b,nir_u2u64(&b,nir_imul_imm(&b,id,16)),0x20d0020000ULL);
 nir_store_global(&b,out,address,.align_mul=16,.write_mask=(1<<count)-1);
 agx_preprocess_nir(b.shader);struct agx_shader_key key={.dev={.chip=AGX_CHIP_G17P,.encoding=encoding},.no_preamble=true};struct agx_shader_part part={0};bool ok=agx_compile_shader_nir(b.shader,&key,&part);
 printf("mode=%u count=%u encoding=%u offset=%d: %s %s\n",mode,count,encoding,offset,ok?"PASS":"REFUSED",ok?"":agx_last_compile_refusal());
 if(ok){assert(part.info.uses_ray_traversal);assert(!part.info.ray_traversal_depth_nonzero);unsigned loads=0,stores=0;for(unsigned i=part.info.main_offset;i+14<=part.info.main_offset+part.info.main_size;i++){struct agx3_scs_desc d;bool l=agx3_scs_decode_load(encoding,(const uint8_t*)part.binary+i,14,&d),s=!l&&agx3_scs_decode_store(encoding,(const uint8_t*)part.binary+i,14,&d);if(l||s){assert(d.byte_offset==offset);assert(d.component_mask==((1<<count)-1));loads+=l;stores+=s;}}assert(loads==1&&stores==1);printf("bytes=%u gprs=%u\n",part.info.main_size,part.info.nr_gprs);}
 free(part.binary);ralloc_free(b.shader);return ok;
}
int main(void){setbuf(stdout,NULL);glsl_type_singleton_init_or_ref();for(unsigned m=0;m<4;m++)for(unsigned c=1;c<=4;c++)assert(check(m,c,AGX_ISA_ENCODING_AGX3B,0x6800));assert(!check(0,1,AGX_ISA_ENCODING_AGX3B,32768));assert(!check(0,1,AGX_ISA_ENCODING_AGX2,0));glsl_type_singleton_decref();return 0;}
