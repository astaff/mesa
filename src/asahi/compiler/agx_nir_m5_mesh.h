/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#pragma once
#include "compiler/nir/nir.h"
#include "agx_mesh_abi.h"
#ifdef __cplusplus
extern "C" {
#endif






bool agx_nir_lower_m5_mesh_triangle_probe(nir_shader *shader);














struct agx_m5_mesh_varyings {
   unsigned size;
   bool dynamic;
   uint8_t primitive_components;
   uint16_t primitive_stride_B;
   uint8_t clip_first, clip_count;
   uint8_t lv_first;
   bool lv;
   uint8_t first[64];
   uint8_t components[64];
   uint8_t primitive_first[64];
   uint8_t primitive_comps[64];
};



bool agx_nir_lower_m5_mesh_fragment(nir_shader *shader);
bool agx_nir_lower_m5_mesh_fragment_layout(nir_shader *shader,
                                           const struct agx_m5_mesh_varyings *mesh);

bool agx_m5_mesh_gather_varyings(const nir_shader *shader,
                                 struct agx_m5_mesh_varyings *varyings);

bool agx_nir_lower_m5_mesh(nir_shader *shader,
                           const struct agx_m5_mesh_varyings *varyings);


bool agx_nir_lower_m5_task_payload_probe(nir_shader *shader);
bool agx_nir_lower_m5_task_probe(nir_shader *shader);
#ifdef __cplusplus
}
#endif
