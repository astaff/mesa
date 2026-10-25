/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#ifndef HK_ACCELERATION_STRUCTURE_H
#define HK_ACCELERATION_STRUCTURE_H

#include "hk_bvh.h"

#include "vk_acceleration_structure.h"

struct hk_device;
struct hk_cmd_buffer;
struct agx_scs_layout_gen2;
struct vk_buffer;
void hk_gen1_publish_buffer_as_addresses(struct hk_device *dev,
                                         const struct vk_buffer *buffer);
uint64_t hk_gen1_upload_as_table(struct hk_cmd_buffer *cmd);
uint64_t hk_gen1_as_metadata_addr(struct hk_cmd_buffer *cmd,
                                  struct vk_acceleration_structure *as);
struct hk_agx3_leaf_args;
struct hk_agx3_envelope_args;
struct hk_agx3_plan_args;

void hk_gen1_rq_finish(struct hk_device *dev);
VkResult hk_device_init_accel_struct_state(struct hk_device *device);
void hk_device_finish_accel_struct_state(struct hk_device *device);
uint64_t hk_m5_upload_as_table(struct hk_cmd_buffer *cmd);
void hk_m5_as_buffer_bound(struct hk_device *dev, struct vk_buffer *buffer);

/* Native build phase; the caller owns job planning, validated output ranges,
 * barriers and meta state save/restore. */
VkResult hk_cmd_encode_agx3_nodes(VkCommandBuffer command_buffer,
                                const struct hk_agx3_leaf_args *args);
VkResult hk_cmd_finalize_agx3_as(VkCommandBuffer command_buffer,
                               const struct hk_agx3_envelope_args *args);
/* Plans, encodes and finalizes one triangle AS from common build IR, with
 * internal phase barriers and a GPU-written indirect node count. */
VkResult hk_cmd_build_agx3_from_ir(VkCommandBuffer command_buffer,
                                 const struct hk_agx3_plan_args *args,
                                 VkDeviceAddress node_status,
                                 VkDeviceAddress envelope_status,
                                 uint32_t slot_bias);

uint64_t hk_gen1_ray_query_scs_layout(const struct hk_device *dev);
uint64_t hk_gen1_query_state(struct hk_cmd_buffer *cmd,
                            const struct agx_scs_layout_gen2 *scs);

#endif /* HK_ACCELERATION_STRUCTURE_H */
