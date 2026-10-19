/*
 * Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 *
 * Opt-in "frame anatomy" profiler: one compact log line per submitted
 * control stream (render or compute) with the hardware's framebuffer and
 * attachment load/store configuration, the barriers exactly as submitted and
 * firmware GPU timestamps.
 *
 * Enabled only by HK_FRAME_ANATOMY=<path> (or "-" for stderr). Every
 * HK_FRAME_ANATOMY_EVERY-th queue submit is sampled (default 200).
 * HK_FRAME_ANATOMY_TS selects which commands of a sampled submit get profiler
 * timestamps: "all" (default), "edges" (first and last hardware command only;
 * the M5 kernel turns every timestamped command into a host-prefix producer,
 * so this is the low-perturbation whole-submit measurement) or "none".
 * When HK_FRAME_ANATOMY is unset, hk_device::frame_anatomy stays NULL and
 * every hook below reduces to one predictable NULL test; no control stream,
 * shader or kernel command is changed.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "vulkan/vulkan_core.h"

struct hk_device;
struct hk_queue;
struct hk_cs;
struct drm_asahi_cmd_render;
struct drm_asahi_cmd_compute;
struct hk_frame_anatomy_submit;

/* Per-render-target load/store summary carried by struct hk_bg_eot, recorded
 * only while the profiler is enabled. Bit 7 marks an attachment with an image
 * view; bits 0-1 hold the op the program actually performs (0 none, 1 clear,
 * 2 load, 3 store); bits 2-3 hold the API loadOp (0 LOAD, 1 CLEAR, 2
 * DONT_CARE, 3 NONE); bits 4-5 hold the API storeOp (0 STORE, 1 DONT_CARE,
 * 2 NONE). */
#define HK_FA_OP_PRESENT 0x80u
#define HK_FA_OP_HW(x) ((x) & 3u)
#define HK_FA_OP_LOAD(x) (((x) >> 2) & 3u)
#define HK_FA_OP_STORE(x) (((x) >> 4) & 3u)

uint8_t hk_frame_anatomy_encode_op(bool present, unsigned hw_op,
                                   VkAttachmentLoadOp load,
                                   VkAttachmentStoreOp store);

void hk_frame_anatomy_init(struct hk_device *dev);
void hk_frame_anatomy_finish(struct hk_device *dev);

/* Returns NULL unless this submit is sampled. */
struct hk_frame_anatomy_submit *
hk_frame_anatomy_begin_submit(struct hk_device *dev, struct hk_queue *queue,
                              unsigned command_count);

/* Record one hardware command of a sampled submit. header_offset is the
 * payload offset at which the command's drm_asahi_cmd_header is appended
 * next. In "all" mode this claims the command's free timestamp slots (never
 * one a query already owns). */
void hk_frame_anatomy_render(struct hk_frame_anatomy_submit *s,
                             struct hk_device *dev, const struct hk_cs *cs,
                             struct drm_asahi_cmd_render *c,
                             unsigned cmdbuf_index, unsigned command_index,
                             size_t header_offset);
void hk_frame_anatomy_compute(struct hk_frame_anatomy_submit *s,
                              struct hk_device *dev, const struct hk_cs *cs,
                              struct drm_asahi_cmd_compute *c,
                              unsigned cmdbuf_index, unsigned command_index,
                              size_t header_offset);

/* Payload complete, before any ioctl: "edges" mode claims the timestamp
 * slots of the first and last recorded command in place. */
void hk_frame_anatomy_prepare(struct hk_frame_anatomy_submit *s,
                              struct hk_device *dev, void *payload,
                              size_t payload_size);

/* Capture each command's header barriers (and exact-prior ordinals) from the
 * payload bytes [begin, end) exactly as they are about to reach the kernel. */
void hk_frame_anatomy_note_barriers(struct hk_frame_anatomy_submit *s,
                                    const void *payload, size_t begin,
                                    size_t end);

/* Queue the records for readback once queue->drm.timeline_value completes;
 * a failed result retires them immediately without timestamps. */
void hk_frame_anatomy_end_submit(struct hk_frame_anatomy_submit *s,
                                 struct hk_device *dev, struct hk_queue *queue,
                                 VkResult result);
