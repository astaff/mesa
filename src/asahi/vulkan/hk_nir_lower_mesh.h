/* Copyright 2026 Aurora Silicon. SPDX-License-Identifier: MIT */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "compiler/nir/nir.h"
#include "compiler/shader_enums.h"
#include "drm-uapi/asahi_drm.h"


enum hk_mesh_mode { HK_MESH_MODE_OFF=0, HK_MESH_MODE_NATIVE_PROBE };
static inline enum hk_mesh_mode hk_mesh_mode(void) {
   const char *mode=getenv("ASAHI_MESH");
   return mode && !strcmp(mode,"native-probe") ? HK_MESH_MODE_NATIVE_PROBE
                                               : HK_MESH_MODE_OFF;
}

/* Opt-in M5 task (object) shaders. ASAHI_M5_TASK=1 advertises taskShader on
 * G17S in native-probe mesh mode. The native object->mesh record (child grid
 * and payload at the SR62/SR63 payload address) needs the kernel to bind the
 * TA mesh rendering buffer (0x1ca48) to task kicks, which it reports with
 * DRM_ASAHI_FEATURE_TASK_MESH_STORAGE. Without it only payload-free
 * task shaders are accepted. */
static inline bool hk_m5_task_opt_in(void) {
   const char *task=getenv("ASAHI_M5_TASK");
   return hk_mesh_mode()==HK_MESH_MODE_NATIVE_PROBE && task && !strcmp(task,"1");
}
static inline bool hk_m5_task_record(uint64_t kernel_features) {
   return hk_m5_task_opt_in() &&
          (kernel_features & DRM_ASAHI_FEATURE_TASK_MESH_STORAGE);
}
static inline unsigned hk_m5_task_payload_limit(uint64_t kernel_features) {
   return hk_m5_task_record(kernel_features) ? 16384 : 0;
}

struct hk_mesh_info {
   uint32_t max_vertices;
   uint32_t max_primitives;
   uint32_t vertex_stride_B;
   uint32_t primitive_stride_B;
   uint32_t local_size[3];

   


   uint64_t vertex_outputs;
   uint64_t primitive_outputs;
   bool writes_cull_primitive;

   

   bool reads_draw_id;
   bool reads_num_workgroups;

   
   uint8_t prim;
   uint8_t clip_distance_array_size;
   uint8_t cull_distance_array_size;

   
   uint32_t payload_size_B;
};
