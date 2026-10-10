/*
 * Copyright © 2021 Google
 * Copyright © 2023 Valve Corporation
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * VK_KHR_acceleration_structure for Honeykrisp.
 *
 * The heavy lifting -- Morton sorting, PLOC/HPLOC or LBVH
 * construction -- is Mesa's common builder, which produces a driver-neutral
 * binary tree in scratch memory.  All this file adds is the conversion of
 * that tree into the node format the software traversal reads (hk_bvh.h),
 * which happens in one compute dispatch per structure, plus the entry points
 * and the handful of vk_device hooks the common builder calls back into.
 */

#include "hk_acceleration_structure.h"

#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"

#include "vk_acceleration_structure.h"
#include "vk_common_entrypoints.h"
#include "vk_meta.h"

#include "bvh/hk_build_interface.h"

#include "util/u_math.h"

static const uint32_t encode_spv[] = {
#include "bvh/hk_encode.spv.h"
};

static const uint32_t write_spv[] = {
#include "bvh/hk_bvh_write.spv.h"
};

static_assert(sizeof(struct hk_bvh_triangle_node) == 48, "node format");
static_assert(sizeof(struct hk_bvh_aabb_node) == 32, "node format");
static_assert(sizeof(struct hk_bvh_instance_node) == 120, "node format");
static_assert(sizeof(struct hk_bvh_box_node) == 56, "node format");
static_assert(sizeof(struct hk_bvh_box_node) == HK_BVH_NODE_PREFETCH_SIZE,
              "the traversal prefetches a whole box node before it knows the "
              "node type, so no node may be larger than one");

static VkResult
get_pipeline_spv(struct hk_device *dev, const char *name, const uint32_t *spv,
                 uint32_t spv_size, unsigned push_constant_size,
                 VkPipeline *pipeline, VkPipelineLayout *layout)
{
   size_t key_size = strlen(name);

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = push_constant_size,
   };

   VkResult result = vk_meta_get_pipeline_layout(
      &dev->vk, &dev->meta, NULL, &pc_range, name, key_size, layout);
   if (result != VK_SUCCESS)
      return result;

   VkPipeline cached = vk_meta_lookup_pipeline(&dev->meta, name, key_size);
   if (cached != VK_NULL_HANDLE) {
      *pipeline = cached;
      return VK_SUCCESS;
   }

   const VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_size,
      .pCode = spv,
   };
   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &module_info,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .pName = "main",
         },
      .layout = *layout,
   };

   return vk_meta_create_compute_pipeline(&dev->vk, &dev->meta, &pipeline_info,
                                          name, key_size, pipeline);
}

/* ---------------------------------------------------------------- */
/* Layout                                                            */

static void
hk_leaf_node_sizes(VkGeometryTypeKHR geometry_type, uint32_t *ir_size,
                   uint32_t *out_size)
{
   switch (geometry_type) {
   case VK_GEOMETRY_TYPE_TRIANGLES_KHR:
      *ir_size = sizeof(struct vk_ir_triangle_node);
      *out_size = sizeof(struct hk_bvh_triangle_node);
      return;
   case VK_GEOMETRY_TYPE_AABBS_KHR:
      *ir_size = sizeof(struct vk_ir_aabb_node);
      *out_size = sizeof(struct hk_bvh_aabb_node);
      return;
   default:
      *ir_size = sizeof(struct vk_ir_instance_node);
      *out_size = sizeof(struct hk_bvh_instance_node);
      return;
   }
}

static VkDeviceSize
hk_as_size(VkGeometryTypeKHR geometry_type, uint32_t leaf_node_count)
{
   /* The common builder always produces MAX2(leaves, 2) - 1 internal nodes,
    * so a one-primitive structure still has a root to point at. */
   uint32_t internal_node_count = MAX2(leaf_node_count, 2) - 1;

   uint32_t ir_size, out_size;
   hk_leaf_node_sizes(geometry_type, &ir_size, &out_size);

   uint32_t nodes_size = internal_node_count * sizeof(struct hk_bvh_box_node) +
                         leaf_node_count * out_size;

   /* Pad so the traversal's unconditional whole-node prefetch of the last
    * node cannot read past the allocation. */
   nodes_size = util_align_npot(nodes_size, HK_BVH_NODE_PREFETCH_SIZE);

   return sizeof(struct hk_bvh_header) + nodes_size;
}

static VkDeviceSize
hk_get_as_size(VkDevice _device,
               const struct vk_acceleration_structure_build_state *state)
{
   return hk_as_size(vk_get_as_geometry_type(state->build_info),
                     state->leaf_node_count);
}

/* ---------------------------------------------------------------- */
/* Encode                                                            */

static void
hk_encode_as(VkCommandBuffer commandBuffer,
             const struct vk_acceleration_structure_build_state *state,
             VkPipelineLayout layout)
{
   VK_FROM_HANDLE(vk_acceleration_structure, dst,
                  state->build_info->dstAccelerationStructure);

   VkGeometryTypeKHR geometry_type =
      vk_get_as_geometry_type(state->build_info);

   VkDeviceAddress scratch = state->build_info->scratchData.deviceAddress;
   uint32_t internal_bound = MAX2(state->leaf_node_count, 2) - 1;

   const struct hk_encode_args args = {
      .intermediate_bvh = scratch + state->scratch.ir_offset,
      .output_bvh = vk_acceleration_structure_get_va(dst),
      .header = scratch + state->scratch.header_offset,
      .root_offset =
         state->scratch.internal_node_offset - state->scratch.ir_offset,
      .internal_node_bound = internal_bound,
      .leaf_node_bound = state->leaf_node_count,
      .geometry_type = geometry_type,
      .as_size = hk_as_size(geometry_type, state->leaf_node_count),
   };
   vk_common_CmdPushConstants(commandBuffer, layout,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(args),
                              &args);

   /* One invocation per output node.  The shader clamps against the counts
    * the builder wrote into the header, which can be smaller than this bound
    * when leaves turn out inactive. */
   hk_cmd_dispatch_unaligned(commandBuffer,
                             internal_bound + state->leaf_node_count, 1, 1);
}

static void
hk_encode(VkCommandBuffer commandBuffer, struct vk_device *device,
          struct vk_meta_device *meta,
          const struct vk_acceleration_structure_build_args *build_args,
          struct vk_acceleration_structure_build_state *states,
          uint32_t build_count, bool flushed_compute_after_init_update_scratch)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   VkPipeline pipeline;
   VkPipelineLayout layout;
   VkResult result =
      get_pipeline_spv(dev, "hk-bvh-encode", encode_spv, sizeof(encode_spv),
                       sizeof(struct hk_encode_args), &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   /* The common builder hands encoding the completed intermediate tree. */
   vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);
   vk_common_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipeline);
   for (uint32_t i = 0; i < build_count; i++)
      hk_encode_as(commandBuffer, &states[i], layout);
}

static const struct vk_acceleration_structure_build_ops hk_as_build_ops = {
   .get_as_size = hk_get_as_size,
   .encode = hk_encode,
};

/* ---------------------------------------------------------------- */
/* vk_device hooks the common builder calls                          */

static void
hk_write_buffer_cp(VkCommandBuffer commandBuffer, VkDeviceAddress addr,
                   void *data, uint32_t size)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   struct {
      uint64_t dst;
      uint32_t count;
      uint32_t data[28];
   } args = {
      .dst = addr,
      .count = DIV_ROUND_UP(size, 4),
   };

   static_assert(sizeof(struct vk_ir_header) <= sizeof(args.data),
                 "common BVH header fits the write shader push constants");

   /* The only caller writes a vk_ir_header, which is 108 bytes.  Anything
    * larger is a change in the common builder, not a bigger buffer: refuse
    * rather than write a truncated header that would fail obscurely later. */
   assert(size <= sizeof(args.data));
   if (size > sizeof(args.data))
      return;
   memcpy(args.data, data, size);

   VkPipeline pipeline;
   VkPipelineLayout layout;
   VkResult result =
      get_pipeline_spv(dev, "hk-bvh-write", write_spv, sizeof(write_spv),
                       sizeof(args), &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   vk_common_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipeline);
   vk_common_CmdPushConstants(commandBuffer, layout,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(args),
                              &args);
   hk_cmd_dispatch_unaligned(commandBuffer, args.count, 1, 1);
}

static void
hk_flush_buffer_write_cp(VkCommandBuffer commandBuffer)
{
   /* Header writes are compute dispatches and must be visible before the
    * common builder's first leaf dispatch. */
   vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);
}

static void
hk_cmd_fill_buffer_addr(VkCommandBuffer commandBuffer, VkDeviceAddress addr,
                        VkDeviceSize size, uint32_t data)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   vk_meta_fill_buffer_addr(&cmd->vk, &dev->meta, addr, size, data);
}

/* ---------------------------------------------------------------- */
/* Device state                                                      */

VkResult
hk_device_init_accel_struct_state(struct hk_device *dev)
{
   dev->accel_struct_args = (struct vk_acceleration_structure_build_args){
      .subgroup_size = 32,
      .bvh_bounds_offset = offsetof(struct hk_bvh_header, bounds),
      /* This software node format does not store propagated cull flags. */
      .root_flags_offset = 0,
      .morton_sort_workgroup_size = 512,
      .morton_sort_kvs_per_thread = 2,
      .propagate_cull_flags = false,
      .emit_markers = false,
   };

   dev->vk.as_build_ops = &hk_as_build_ops;
   dev->vk.write_buffer_cp = hk_write_buffer_cp;
   dev->vk.flush_buffer_write_cp = hk_flush_buffer_write_cp;
   dev->vk.cmd_dispatch_unaligned = hk_cmd_dispatch_unaligned;
   dev->vk.cmd_fill_buffer_addr = hk_cmd_fill_buffer_addr;

   return VK_SUCCESS;
}

/* ---------------------------------------------------------------- */
/* Entry points                                                      */

VKAPI_ATTR void VKAPI_CALL
hk_GetAccelerationStructureBuildSizesKHR(
   VkDevice _device, VkAccelerationStructureBuildTypeKHR buildType,
   const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo,
   const uint32_t *pMaxPrimitiveCounts,
   VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo)
{
   VK_FROM_HANDLE(hk_device, dev, _device);

   vk_get_as_build_sizes(_device, buildType, pBuildInfo, pMaxPrimitiveCounts,
                         pSizeInfo, &dev->accel_struct_args);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBuildAccelerationStructuresKHR(
   VkCommandBuffer commandBuffer, uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* A build binds its own compute pipelines and push constants, so the
    * application's compute state has to survive it. */
   struct hk_meta_save save;
   hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);

   vk_cmd_build_acceleration_structures(commandBuffer, &dev->vk, &dev->meta,
                                        infoCount, pInfos, ppBuildRangeInfos,
                                        &dev->accel_struct_args);

   hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDeviceAccelerationStructureCompatibilityKHR(
   VkDevice _device, const VkAccelerationStructureVersionInfoKHR *pVersionInfo,
   VkAccelerationStructureCompatibilityKHR *pCompatibility)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   struct hk_physical_device *pdev = hk_device_physical(dev);

   bool compat = memcmp(pVersionInfo->pVersionData,
                        pdev->vk.properties.driverUUID, VK_UUID_SIZE) == 0;
   *pCompatibility = compat
                        ? VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR
                        : VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}

/*
 * Host-side builds and the copy/serialize family are unimplemented and, in
 * the case of the host builds, unadvertised: accelerationStructureHostCommands
 * stays false, so a conformant application never reaches them.  They fail
 * loudly rather than silently returning success on an untouched structure.
 */
VKAPI_ATTR VkResult VKAPI_CALL
hk_BuildAccelerationStructuresKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CopyAccelerationStructureKHR(VkDevice _device,
                                VkDeferredOperationKHR deferredOperation,
                                const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CopyMemoryToAccelerationStructureKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CopyAccelerationStructureToMemoryKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_WriteAccelerationStructuresPropertiesKHR(
   VkDevice _device, uint32_t accelerationStructureCount,
   const VkAccelerationStructureKHR *pAccelerationStructures,
   VkQueryType queryType, size_t dataSize, void *pData, size_t stride)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}
