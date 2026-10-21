/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <vulkan/vulkan_core.h>

struct hk_host_as_staging {
   uint64_t vertices, indices, transform, data;
   uint32_t vertex_count;
};


bool hk_host_as_stage_layout(const VkAccelerationStructureGeometryKHR *g,
   const VkAccelerationStructureBuildRangeInfoKHR *r,
   struct hk_host_as_staging *layout, uint64_t *offset);
void hk_host_as_stage_geometry(VkAccelerationStructureGeometryKHR *g,
   VkAccelerationStructureBuildRangeInfoKHR *r,
   const struct hk_host_as_staging *layout, uint8_t *map, uint64_t va);
