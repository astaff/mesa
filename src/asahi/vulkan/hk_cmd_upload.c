/* SPDX-License-Identifier: MIT */
#include <inttypes.h>
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_shader.h"
#include "util/u_cpu_detect.h"

static struct agx_ptr
hk_pool_root_alloc(struct hk_cmd_buffer *cmd, uint32_t size,
                   uint32_t alignment)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!agx_native_graphics(&dev->dev))
      return hk_pool_alloc(cmd, size, alignment);

   return hk_pool_alloc_from(cmd, &cmd->uploader.root, size, alignment,
                             HK_CMD_BO_ROOT);
}

__attribute__((hot)) uint64_t
hk_cmd_buffer_upload_root(struct hk_cmd_buffer *cmd,
                          VkPipelineBindPoint bind_point)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_descriptor_state *desc = hk_get_descriptors_state(cmd, bind_point);
   struct hk_root_descriptor_table *root = &desc->root;

   struct agx_ptr root_ptr = hk_pool_root_alloc(cmd, sizeof(*root), 8);

   /* A HAL200 tail built for the old root names it and must be rebuilt
    * before publication. */
   if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS &&
       cmd->state.gfx.root && root_ptr.gpu != cmd->state.gfx.root)
      cmd->state.gfx.g17p_bind_tail.valid = false;
   if (!root_ptr.gpu)
      return 0;

   root->root_desc_addr = root_ptr.gpu;
   root->printf_buffer_address = dev->dev.printf_buffer_address;
   root->sampler_heap_address =
      dev->samplers.table.bo && dev->samplers.table.bo->va
         ? dev->samplers.table.bo->va->addr
         : 0;

   /* Dynamic buffers past the highest one ever bound are read by no bound
    * layout: copy the table around them. */
   const size_t head_B = hk_root_descriptor_offset(dynamic_buffers) +
                         desc->dynamic_buffer_end * sizeof(root->dynamic_buffers[0]);
   const size_t tail_B = hk_root_descriptor_offset(set_dynamic_buffer_start);
   memcpy(root_ptr.cpu, root, head_B);
   memcpy((uint8_t *)root_ptr.cpu + tail_B, (const uint8_t *)root + tail_B,
          sizeof(*root) - tail_B);

   /* Cached G17P roots need explicit CPU-to-GPU publication. Clean just this
    * table now; vkEndCommandBuffer completes all recorded root cleans with a
    * single fence before submission can consume them. */
   if (agx_native_graphics(&dev->dev) && util_has_cache_ops()) {
      util_flush_range_no_fence(root_ptr.cpu, head_B);
      util_flush_range_no_fence((uint8_t *)root_ptr.cpu + tail_B,
                                sizeof(*root) - tail_B);
   }

   hk_upload_stat(cmd, root_uploads, 1);
   hk_upload_stat(cmd, root_bytes, head_B + sizeof(*root) - tail_B);
   return root_ptr.gpu;
}

uint64_t
hk_cmd_upload_constants(struct hk_cmd_buffer *cmd, const struct hk_shader *s,
                        UNUSED mesa_shader_stage stage)
{
   if (!s || !s->data_size)
      return 0;

   uint64_t addr = hk_pool_upload(cmd, s->data_ptr, s->data_size,
                                  HK_MIN_UBO_ALIGNMENT);
   if (!addr)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   else {
      hk_upload_stat(cmd, constant_uploads, 1);
      hk_upload_stat(cmd, constant_bytes, s->data_size);
   }
   return addr;
}

void
hk_cmd_upload_draw_params(struct hk_cmd_buffer *cmd, uint32_t first,
                          uint32_t first_instance)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   uint32_t params[] = {first, first_instance};
   memcpy(gfx->g17p_draw_params_value, params, sizeof(params));
   gfx->g17p_draw_params_host_valid = true;
   gfx->draw_params = hk_pool_upload(cmd, params, sizeof(params), 4);
   hk_upload_stat(cmd, draw_param_uploads, 1);
   hk_upload_stat(cmd, draw_param_bytes, sizeof(params));
}

void
hk_cmd_upload_draw_id(struct hk_cmd_buffer *cmd, uint32_t draw_id)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   gfx->g17p_draw_id_value = draw_id;
   gfx->g17p_draw_id_host_valid = true;
   /* Native uniform DMA reads a whole dword, including the high half. */
   gfx->draw_id_ptr = hk_pool_upload(cmd, &draw_id, sizeof(draw_id), 4);
   hk_upload_stat(cmd, draw_id_uploads, 1);
   hk_upload_stat(cmd, draw_id_bytes, sizeof(draw_id));
}

void
hk_cmd_print_upload_stats(struct hk_cmd_buffer *cmd)
{
   const struct hk_upload_stats *s = &cmd->state.upload_stats;
   perf_debug(cmd, "CPU uploads: draws=%" PRIu64
              " descriptor_binds=%" PRIu64 " descriptor_noops=%" PRIu64
              " root=%" PRIu64 "/%" PRIu64 "B root_reuse=%" PRIu64
              " draw_params=%" PRIu64 "/%" PRIu64 "B draw_param_reuse=%" PRIu64
              " draw_id=%" PRIu64 "/%" PRIu64 "B draw_id_reuse=%" PRIu64
              " constants=%" PRIu64 "/%" PRIu64 "B constant_reuse=%" PRIu64,
              s->draws, s->descriptor_binds, s->descriptor_noops,
              s->root_uploads, s->root_bytes, s->root_reuses,
              s->draw_param_uploads, s->draw_param_bytes, s->draw_param_reuses,
              s->draw_id_uploads, s->draw_id_bytes, s->draw_id_reuses,
              s->constant_uploads, s->constant_bytes, s->constant_reuses);
}
