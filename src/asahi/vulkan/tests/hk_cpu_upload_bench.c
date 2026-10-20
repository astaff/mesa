/* SPDX-License-Identifier: MIT
 * Host-only command-recording microbenchmark. No Vulkan device is created and
 * nothing is submitted. The real descriptor-bind and upload paths run against
 * preallocated command-buffer storage. Run with ASAHI_MESA_DEBUG=perf.
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hk_cmd_buffer.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "vk_pipeline_layout.h"

#define CHECK(c) do { if (!(c)) { \
   fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #c); \
   exit(EXIT_FAILURE); } } while (0)

struct fixture {
   struct hk_cmd_buffer cmd;
   struct hk_device dev;
   struct vk_pipeline_layout layout;
   struct hk_descriptor_set_layout set_layout;
   /* hk_descriptor_set ends in a dynamic-buffer array. */
   struct hk_descriptor_set *set;
   struct hk_shader vs, fs;
   struct hk_api_shader *vs_api, *fs_api;
   uint8_t constants[128];
   void *main_map, *root_map;
};

static void
reset(struct fixture *f)
{
   memset(&f->cmd.state, 0, sizeof(f->cmd.state));
   f->cmd.uploader.main.offset = f->cmd.uploader.root.offset = 0;
   f->cmd.state.gfx.shaders[MESA_SHADER_VERTEX] = f->vs_api;
   f->cmd.state.gfx.shaders[MESA_SHADER_FRAGMENT] = f->fs_api;
}

static struct fixture *
fixture(void)
{
   struct fixture *f = calloc(1, sizeof(*f));
   CHECK(f);
   f->main_map = calloc(1, HK_CMD_BO_SIZE);
   f->root_map = calloc(1, HK_CMD_BO_SIZE);
   f->set = calloc(1, sizeof(*f->set) + sizeof(f->set->dynamic_buffers[0]));
   CHECK(f->main_map && f->root_map && f->set);
   f->dev.dev.chip = AGX_CHIP_G17P;
   f->dev.dev.debug = AGX_DBG_PERF;
   f->cmd.vk.base.device = &f->dev.vk;
   f->cmd.vk.base.type = VK_OBJECT_TYPE_COMMAND_BUFFER;
   f->cmd.uploader.main.map = f->main_map;
   f->cmd.uploader.main.base = 0x100000;
   f->cmd.uploader.root.map = f->root_map;
   f->cmd.uploader.root.base = 0x200000;
   f->layout.base.type = VK_OBJECT_TYPE_PIPELINE_LAYOUT;
   f->set_layout.vk.base.type = VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT;
   f->set_layout.vk.dynamic_descriptor_count = 1;
   f->layout.set_layouts[0] = &f->set_layout.vk;
   f->set->base.type = VK_OBJECT_TYPE_DESCRIPTOR_SET;
   f->set->layout = &f->set_layout;
   f->set->addr = 0x300000;
   f->set->size = 64;
   f->set->dynamic_buffers[0].base_addr = 0x400000;
   f->set->dynamic_buffers[0].size = 4096;
   f->vs.data_ptr = f->fs.data_ptr = f->constants;
   f->vs.data_size = f->fs.data_size = sizeof(f->constants);
   f->vs.constant_data_id = 1;
   f->fs.constant_data_id = 2;
   f->vs_api = calloc(1, hk_api_shader_size(MESA_SHADER_VERTEX, false));
   f->fs_api = calloc(1, hk_api_shader_size(MESA_SHADER_FRAGMENT, false));
   CHECK(f->vs_api && f->fs_api);
   f->vs_api->vk.stage = MESA_SHADER_VERTEX;
   f->fs_api->vk.stage = MESA_SHADER_FRAGMENT;
   f->vs_api->variants[HK_VS_VARIANT_HW].compiled = true;
   f->fs_api->variants[0].compiled = true;
   reset(f);
   CHECK(16 * sizeof(struct hk_root_descriptor_table) <= HK_CMD_BO_SIZE);
   return f;
}

static void
bind(struct fixture *f, uint32_t offset, VkShaderStageFlags stages)
{
   VkDescriptorSet set = hk_descriptor_set_to_handle(f->set);
   VkBindDescriptorSetsInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO_KHR,
      .stageFlags = stages,
      .layout = vk_pipeline_layout_to_handle(&f->layout),
      .descriptorSetCount = 1,
      .pDescriptorSets = &set,
      .dynamicOffsetCount = 1,
      .pDynamicOffsets = &offset,
   };
   hk_CmdBindDescriptorSets2KHR(hk_cmd_buffer_to_handle(&f->cmd), &info);
}

static void
correctness(struct fixture *f)
{
   struct hk_cmd_buffer *cmd = &f->cmd;
   bind(f, 16, VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT);
   CHECK(cmd->state.gfx.descriptors.root.dynamic_buffers[0].base_addr == 0x400010);
   CHECK(cmd->state.cs.descriptors.root.dynamic_buffers[0].base_addr == 0x400010);
   uint64_t root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   CHECK(root);
   uint8_t old[sizeof(struct hk_root_descriptor_table)];
   memcpy(old, f->root_map, sizeof(old));
   bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
   CHECK(cmd->state.gfx.descriptors.root.dynamic_buffers[0].base_addr == 0x400020);
   CHECK(cmd->state.cs.descriptors.root.dynamic_buffers[0].base_addr == 0x400010);
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS) != root);
   CHECK(memcmp(old, f->root_map, sizeof(old)) == 0);
   /* Switching from push descriptors back to the same set is a real bind. */
   cmd->state.gfx.descriptors.push_active = 1;
   cmd->state.gfx.descriptors.root.sets[0] = 0x500000;
   bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
   CHECK(!cmd->state.gfx.descriptors.push_active);
   CHECK(cmd->state.gfx.descriptors.root.sets[0] == f->set->addr);
   /* An identical set with changed dynamic-layout indexing is a real change. */
   f->layout.dynamic_descriptor_offset[0] = 1;
   bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
   CHECK(cmd->state.gfx.descriptors.root.set_dynamic_buffer_start[0] == 1);
   CHECK(cmd->state.gfx.descriptors.root.dynamic_buffers[1].base_addr == 0x400020);
   f->layout.dynamic_descriptor_offset[0] = 0;
   hk_cmd_upload_draw_params(cmd, (uint32_t)-7, 3);
   uint64_t params = cmd->state.gfx.draw_params;
   const uint32_t *data = (uint32_t *)((uint8_t *)f->main_map + params - 0x100000);
   CHECK(data[0] == (uint32_t)-7 && data[1] == 3);
   /* Indirect parameters are read at execution time; never compare GPU data. */
   cmd->state.gfx.draw_params = 0x600000;
   cmd->state.gfx.g17p_draw_params_host_valid = false;
   hk_cmd_upload_draw_params(cmd, (uint32_t)-7, 3);
   CHECK(cmd->state.gfx.draw_params != 0x600000);
   hk_cmd_upload_draw_params(cmd, 9, 4);
   CHECK(data[0] == (uint32_t)-7 && data[1] == 3);
   hk_cmd_upload_draw_id(cmd, 65535);
   const uint32_t *id = (uint32_t *)((uint8_t *)f->main_map +
                                    cmd->state.gfx.draw_id_ptr - 0x100000);
   CHECK(*id == 65535);
   hk_cmd_upload_draw_id(cmd, 0);
   CHECK(*id == 65535);
   CHECK(cmd->vk.record_result == VK_SUCCESS);
   reset(f);
}

static uint64_t
now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static void
run(struct fixture *f, const char *name, unsigned change_every, unsigned batches)
{
   struct hk_upload_stats totals = {0};
   const uint64_t start = now_ns();
   for (unsigned b = 0; b < batches; ++b) {
      reset(f);
      for (unsigned d = 0; d < 16; ++d) {
         const uint32_t value = d / change_every;
         bind(f, value * 16, VK_SHADER_STAGE_ALL_GRAPHICS);
         struct hk_cmd_buffer *cmd = &f->cmd;
         /* The general draw path calls this only for a dirty root. Force the
          * call too, to measure pass-boundary/over-invalidation reuse. */
         cmd->state.gfx.root =
            hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
         hk_cmd_upload_draw_params(cmd, value, 0);
         hk_cmd_upload_draw_id(cmd, value);
         CHECK(hk_cmd_upload_constants(cmd, &f->vs, MESA_SHADER_VERTEX));
         CHECK(hk_cmd_upload_constants(cmd, &f->fs, MESA_SHADER_FRAGMENT));
         hk_upload_stat(cmd, draws, 1);
      }
#define SUM(field) totals.field += f->cmd.state.upload_stats.field
      SUM(draws); SUM(descriptor_binds); SUM(descriptor_noops);
      SUM(root_uploads); SUM(root_bytes); SUM(root_reuses);
      SUM(draw_param_uploads); SUM(draw_param_bytes); SUM(draw_param_reuses);
      SUM(draw_id_uploads); SUM(draw_id_bytes); SUM(draw_id_reuses);
      SUM(constant_uploads); SUM(constant_bytes); SUM(constant_reuses);
#undef SUM
      CHECK(f->cmd.vk.record_result == VK_SUCCESS);
   }
   const uint64_t elapsed = now_ns() - start;
   f->cmd.state.upload_stats = totals;
   printf("%s batches=%u draws=%" PRIu64 " ns_per_draw=%.2f\n", name, batches,
          totals.draws, (double)elapsed / totals.draws);
   hk_cmd_print_upload_stats(&f->cmd);
   printf("generation_per_batch=%" PRIu64 " root_struct_bytes=%zu\n",
          f->cmd.state.gfx.descriptors.generation,
          sizeof(struct hk_root_descriptor_table));
}

int
main(int argc, char **argv)
{
   unsigned batches = argc > 1 ? strtoul(argv[1], NULL, 10) : 1000;
   CHECK(batches && batches <= 1000000);
   struct fixture *f = fixture();
   correctness(f);
   run(f, "identical", 16, batches);
   run(f, "change_every_4", 4, batches);
   run(f, "change_every_draw", 1, batches);
   free(f->vs_api);
   free(f->fs_api);
   free(f->set);
   free(f->main_map);
   free(f->root_map);
   free(f);
   return 0;
}
