/* Copyright 2026 Aurora Silicon. SPDX-License-Identifier: MIT */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "compiler/nir/nir.h"
#include "compiler/shader_enums.h"


enum hk_mesh_mode { HK_MESH_MODE_OFF=0, HK_MESH_MODE_NATIVE_PROBE };
static inline enum hk_mesh_mode hk_mesh_mode(void) {
   const char *mode=getenv("ASAHI_MESH");
   return mode && !strcmp(mode,"native-probe") ? HK_MESH_MODE_NATIVE_PROBE
                                               : HK_MESH_MODE_OFF;
}

/* Opt-in M5 task (object) shaders. ASAHI_M5_TASK=1 advertises taskShader on
 * G17S in native-probe mesh mode. Task payloads use the native SR62/SR63
 * payload address, which needs kernel backing (the 0x1ca48 mesh rendering
 * buffer); ASAHI_M5_TASK_PAYLOAD gives the payload byte limit to accept
 * (default 0: every payload access is refused). */
static inline bool hk_m5_task_opt_in(void) {
   const char *task=getenv("ASAHI_M5_TASK");
   return hk_mesh_mode()==HK_MESH_MODE_NATIVE_PROBE && task && !strcmp(task,"1");
}
static inline unsigned hk_m5_task_payload_limit(void) {
   const char *limit=getenv("ASAHI_M5_TASK_PAYLOAD");
   if (!hk_m5_task_opt_in() || !limit)
      return 0;
   const unsigned long bytes=strtoul(limit,NULL,0);
   return bytes>16384 ? 16384 : (unsigned)bytes;
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
