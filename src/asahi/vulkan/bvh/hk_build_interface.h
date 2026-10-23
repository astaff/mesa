/*
 * Copyright © 2022 Konstantin Seurer
 * Copyright © 2024 Valve Corporation
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Push-constant blocks shared between the Honeykrisp acceleration-structure
 * encode shaders and the C code that dispatches them.  Everything the
 * encoder needs travels in push constants; there are no descriptor sets, so
 * a build never disturbs the application's bound descriptors.
 */

#ifndef HK_BVH_BUILD_INTERFACE_H
#define HK_BVH_BUILD_INTERFACE_H

#ifdef VULKAN
/* The Neo base uses vk_bvh_helpers for the same fixed IR nodes. */
#define VK_USED_BUILD_FLAGS 0
#include "vk_bvh_helpers.h"
#include "hk_bvh.h"

TYPE(hk_bvh_triangle_node, 4);
TYPE(hk_bvh_aabb_node, 4);
TYPE(hk_bvh_instance_node, 8);
TYPE(hk_bvh_box_node, 4);
TYPE(hk_bvh_header, 4);
#else
#include <stdint.h>

#include "hk_bvh.h"

#define REF(type) uint64_t
#define VOID_REF  uint64_t
#endif

#include "../../libagx/gen1_as_metadata.h"
#ifdef VULKAN
TYPE(libagx_gen1_as_metadata, 8);
#endif

/* Retained binary topology for native refits. Only internal nodes and header
 * persist; updated leaf data is read directly from application buffers.
 *
 * After them, at HK_AGX3_RETAINED_PLAN_OFFSET, an updatable structure keeps
 * the native plan of its last full build: the plan state, then one record
 * per encoding job {IR node index, root slot, child slot}. An update keeps
 * the topology, so it restores this plan instead of planning again: refit
 * phase 5 or 6 copies it to scratch and plan phase 4 rebuilds each job from
 * its IR node with the same bounded walk as emit (no root walks or atomics).
 * The walk's leaf counts are kept in the retained internal nodes' bvh_offset
 * field, which nothing else reads after the common build. */
struct hk_agx3_refit_args {
   VOID_REF retained;
   VOID_REF metadata;
   VOID_REF intermediate_bvh;
   REF(vk_ir_header) header;
   VOID_REF parents;
   uint32_t root_offset;
   uint32_t leaf_bound;
   uint32_t node_bound;
   uint32_t phase;
   /* Phases 4..6: the plan in encode scratch. Phases 5 and 6 leave the job
    * records in the plan-node array, which an update does not use. */
   VOID_REF plan_nodes;
   VOID_REF plan_jobs;
   VOID_REF plan_state;
};

#define HK_AGX3_RETAINED_JOB_WORDS 3u
#define HK_AGX3_RETAINED_PLAN_STATE_BYTES 32u
/* Retained-plan offset from the retained base: past the IR header and the
 * internal-node bound, 16-byte aligned. */
#define HK_AGX3_RETAINED_PLAN_OFFSET(header_bytes, box_bytes, internal_bound) \
   (((header_bytes) + (internal_bound) * (box_bytes) + 15u) / 16u * 16u)
#define HK_AGX3_RETAINED_PLAN_BYTES(jobs) \
   (HK_AGX3_RETAINED_PLAN_STATE_BYTES + (jobs) * (HK_AGX3_RETAINED_JOB_WORDS * 4u))


struct hk_encode_args {
   /* Base of the common builder's intermediate representation: the packed
    * leaf array, immediately followed by the internal box-node array. */
   VOID_REF intermediate_bvh;
   /* Base of the acceleration structure we are writing. */
   VOID_REF output_bvh;
   REF(vk_ir_header) header;
   /* Byte offset of the internal box-node array inside intermediate_bvh,
    * i.e. leaf_node_count * the IR leaf size. */
   uint32_t root_offset;
   /* Upper bound the dispatch was sized from; the shader still clamps
    * against the counts the builder wrote into the header. */
   uint32_t internal_node_bound;
   uint32_t leaf_node_bound;
   uint32_t geometry_type;
   /* What get_as_size() reserved, recorded into the header so a compacting
    * copy can find it without recomputing. */
   uint32_t as_size;
};

/* Native Gen3 node encoder. Jobs refer to Mesa's common triangle/box IR,
 * so the build planner can select/reorder leaves without copying vertices.
 * Each job owns disjoint root and child regions. Slots are relative to its
 * output AS, not the job or the root. The caller supplies the AS metadata.
 * This covers inner nodes and the opaque, geometry-zero 1..8 leaf form;
 * it is not a complete acceleration-structure build implementation. */
struct hk_agx3_leaf_job {
   VOID_REF intermediate_bvh;
   VOID_REF output_bvh;
   uint32_t root_slot;
   uint32_t child_slot;
   uint32_t child_count;
   uint32_t node_kind;
   uint32_t triangle_offsets[8];
};

#ifdef VULKAN
TYPE(hk_agx3_leaf_job, 8);
#endif

struct hk_agx3_leaf_args {
   REF(hk_agx3_leaf_job) jobs;
   REF(uint32_t) status;
   uint32_t job_count;
   /* Slots from the traversal base to output_bvh, added to every child
    * reference (see libagx/gen1_as.h). Zero when output_bvh is the base. */
   uint32_t slot_bias;
};

#define HK_AGX3_LEAF_SUCCESS 0
#define HK_AGX3_LEAF_INVALID_COUNT 1
#define HK_AGX3_LEAF_NONFINITE 2
#define HK_AGX3_LEAF_RANGE 4
#define HK_AGX3_LEAF_UNSUPPORTED_GEOMETRY 8
#define HK_AGX3_NODE_LEAF 0
#define HK_AGX3_NODE_INNER 1
#define HK_AGX3_NODE_INSTANCES 2
#define HK_AGX3_NODE_AABBS 3

/* Finish a single opaque triangle geometry after the node jobs. Bounds name
 * a vk_ir_node from the common builder; metadata offsets remain AS-relative. */
struct hk_agx3_envelope_job {
   REF(vk_ir_node) bounds;
   VOID_REF output_bvh;
   uint32_t end_slot;
   uint32_t primitive_count;
   uint32_t node_count;
   uint32_t depth;
   uint32_t opaque_build_word;
   uint32_t resource_id;
   uint32_t geometry_type;
   uint32_t leaf_bound;
};

#ifdef VULKAN
TYPE(hk_agx3_envelope_job, 8);
#endif

struct hk_agx3_envelope_args {
   REF(hk_agx3_envelope_job) jobs;
   REF(uint32_t) status;
   uint32_t job_count;
   uint32_t reserved;
   VOID_REF metadata;
   REF(vk_ir_header) ir_header;
   uint64_t retained_offset;
   uint32_t build_flags;
   uint32_t node_bound;
   uint32_t keep_retained;
   uint32_t padding;
};

/* Scratch records for parallel binary-to-wide planning. Phases are separate
 * dispatches with storage barriers; no workgroup waits for another to run. */
struct hk_agx3_plan_node {
   uint32_t parent;
   uint32_t leaf_count; /* saturated at nine */
   uint32_t depth;
   uint32_t job;
};

struct hk_agx3_plan_state {
   /* First three words are VkDispatchIndirectCommand for node encoding. */
   uint32_t job_count;
   uint32_t groups_y;
   uint32_t groups_z;
   uint32_t next_slot;
   uint32_t depth;
   uint32_t error;
   uint32_t reserved[2];
};

#ifdef VULKAN
TYPE(hk_agx3_plan_node, 4);
TYPE(hk_agx3_plan_state, 4);
#endif

struct hk_agx3_plan_args {
   VOID_REF intermediate_bvh;
   REF(vk_ir_header) header;
   REF(hk_agx3_plan_node) nodes;
   REF(hk_agx3_leaf_job) jobs;
   REF(hk_agx3_envelope_job) envelope;
   REF(hk_agx3_plan_state) state;
   VOID_REF output_bvh;
   uint32_t root_offset;
   uint32_t leaf_bound;
   uint32_t node_bound;
   uint32_t slot_bound;
   uint32_t phase;
   uint32_t resource_id;
   uint32_t geometry_type;
   /* Phases 0..3 were replaced by a restored plan (an update): phase 4
    * rebuilds the jobs from the records in nodes. */
   uint32_t plan_restored;
   VOID_REF metadata;
   uint64_t retained_offset;
   uint32_t build_flags;
   uint32_t keep_retained;
};

#define HK_AGX3_PLAN_INDEX 1
#define HK_AGX3_PLAN_CYCLE 2
#define HK_AGX3_PLAN_ALLOCATION 4
#define HK_AGX3_PLAN_SHAPE 8
#define HK_AGX3_PLAN_EMPTY 16

#endif /* HK_BVH_BUILD_INTERFACE_H */
