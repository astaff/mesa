/* Copyright 2026 Aurora Silicon. SPDX-License-Identifier: MIT */
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "agx_chip.h"
#ifdef __cplusplus
extern "C" {
#endif

#define AGX_VDM_MESH_STATE_LENGTH    56u
#define AGX_VDM_MESH_DISPATCH_LENGTH 28u
#define AGX_VDM_MESH_STATE_HEADER    UINT32_C(0x40000d2c)
#define AGX_VDM_MESH_DISPATCH_HEADER UINT32_C(0x70000600)


struct agx_vdm_mesh_state_info {
   uint32_t raw_prefix[4]; 
   uint64_t object_pipeline, mesh_pipeline;
   uint32_t raw_object_template, raw_mesh_template;
   uint32_t raw_tail[5]; 
};
struct agx_vdm_mesh_dispatch_info {
   uint32_t threads[3], local_size[3];
};

bool agx_vdm_mesh_threadgroup_pack(const uint32_t size[3], uint32_t *word);
bool agx_vdm_mesh_threadgroup_unpack(uint32_t word, uint32_t size[3]);
bool agx_vdm_mesh_state_pack(enum agx_chip chip,
                             const struct agx_vdm_mesh_state_info *info,
                             uint8_t out[AGX_VDM_MESH_STATE_LENGTH]);
bool agx_vdm_mesh_state_unpack(enum agx_chip chip, const void *bytes,
                               size_t size,
                               struct agx_vdm_mesh_state_info *info);
bool agx_vdm_mesh_dispatch_pack(enum agx_chip chip,
                                const struct agx_vdm_mesh_dispatch_info *info,
                                uint8_t out[AGX_VDM_MESH_DISPATCH_LENGTH]);
bool agx_vdm_mesh_dispatch_unpack(enum agx_chip chip, const void *bytes,
                                  size_t size,
                                  struct agx_vdm_mesh_dispatch_info *info);

#ifdef __cplusplus
}
#endif
