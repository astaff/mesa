/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl.h"

KERNEL(32)
libagx_fill(global uint32_t *address, uint32_t value)
{
   assert((((uintptr_t)address) & 0x3) == 0);

   address[cl_global_id.x] = value;
}

KERNEL(32)
libagx_copy_uint4(global uint4 *dest, global uint4 *src)
{
   assert((((uintptr_t)dest) & 0xf) == 0);
   assert((((uintptr_t)src) & 0xf) == 0);

   dest[cl_global_id.x] = src[cl_global_id.x];
}

KERNEL(16)
libagx_copy_uchar(global uint8_t *dest, global uint8_t *src)
{
   dest[cl_global_id.x] = src[cl_global_id.x];
}

static uint
libagx_space_bits(uint x)
{
   uint accum = 0;
   for (uint i = 0; i < 14; ++i)
      accum |= (x & (1u << i)) << i;
   return accum;
}

static uint
libagx_space_bits_7(uint x)
{
   uint accum = 0;
   for (uint i = 0; i < 7; ++i)
      accum |= (x & (1u << i)) << i;
   return accum;
}

/*
 * Raw texture copies operate on format elements, not interpreted
 * colours.  An element is a pixel for ordinary formats and a whole block for
 * BC formats.  Keeping both sides in element coordinates also implements the
 * specified BC<->RGBA32 copy compatibility without format conversion: one
 * 16-byte BC block is copied to or from one 16-byte RGBA32 pixel.
 *
 * CPU command recording validates every rectangle against the logical image
 * or placed footprint before this kernel is dispatched.  The coordinate
 * guard is a second line of defence against an over-wide launch.
 */
static uint
libagx_copy_image_offset(uint x, uint y, uint pitch_B, uint stride_el,
                         uint tile_width_el, uint tile_height_el,
                         uint blocksize_B, uint tiled)
{
   if (!tiled)
      return y * pitch_B + x * blocksize_B;

   uint tiles_per_row = (stride_el + tile_width_el - 1) / tile_width_el;
   uint tile_idx = (y / tile_height_el) * tiles_per_row +
                   (x / tile_width_el);
   uint tile_area_el = tile_width_el * tile_height_el;
   uint morton_el = libagx_space_bits(x & (tile_width_el - 1)) |
                    (libagx_space_bits(y & (tile_height_el - 1)) << 1);

   return (tile_idx * tile_area_el + morton_el) * blocksize_B;
}

/* Reduce power-of-two tile geometry on the CPU. Small tiles can use the
 * seven-bit encoder, but legacy AIL mip tails may use larger square tiles.
 * Keep their high coordinate bits just as the generic helper does. */
static uint
libagx_copy_image_offset_pow2(uint x, uint y, uint pitch_B,
                              uint tiles_per_row, uint tile_width_log2,
                              uint tile_height_log2, uint blocksize_B,
                              uint tiled)
{
   if (!tiled)
      return y * pitch_B + x * blocksize_B;

   uint tile_width_el = 1u << tile_width_log2;
   uint tile_height_el = 1u << tile_height_log2;
   uint tile_idx = (y >> tile_height_log2) * tiles_per_row +
                   (x >> tile_width_log2);
   uint tile_area_el = 1u << (tile_width_log2 + tile_height_log2);
   uint tile_x = x & (tile_width_el - 1);
   uint tile_y = y & (tile_height_el - 1);
   uint morton_el = tile_width_log2 <= 7 && tile_height_log2 <= 7
      ? libagx_space_bits_7(tile_x) | (libagx_space_bits_7(tile_y) << 1)
      : libagx_space_bits(tile_x) | (libagx_space_bits(tile_y) << 1);

   return (tile_idx * tile_area_el + morton_el) * blocksize_B;
}

KERNEL(32)
libagx_copy_image(global uint8_t *dest, global uint8_t *src,
                  uint width_el, uint height_el,
                  uint dest_x_el, uint dest_y_el,
                  uint src_x_el, uint src_y_el,
                  uint dest_pitch_B, uint src_pitch_B,
                  uint dest_tiles_per_row, uint src_tiles_per_row,
                  uint dest_tile_width_log2, uint dest_tile_height_log2,
                  uint src_tile_width_log2, uint src_tile_height_log2,
                  uint blocksize_B, uint dest_tiled, uint src_tiled)
{
   uint x = cl_global_id.x;
   uint y = cl_global_id.y;
   if (x >= width_el || y >= height_el)
      return;

   uint dest_offset_B = libagx_copy_image_offset_pow2(
      dest_x_el + x, dest_y_el + y, dest_pitch_B, dest_tiles_per_row,
      dest_tile_width_log2, dest_tile_height_log2, blocksize_B, dest_tiled);
   uint src_offset_B = libagx_copy_image_offset_pow2(
      src_x_el + x, src_y_el + y, src_pitch_B, src_tiles_per_row,
      src_tile_width_log2, src_tile_height_log2, blocksize_B, src_tiled);

   for (uint b = 0; b < blocksize_B; ++b)
      dest[dest_offset_B + b] = src[src_offset_B + b];
}

/* Round an unsigned four-sample average like G17P's linear sampler followed
 * by a normalized colour write. Exact output ties move toward code 128. This
 * avoids the still-unqualified G17P float-to-integer convert encoding. */
static uint8_t
libagx_average4_unorm8(uint a, uint b, uint c, uint d)
{
   uint sum = a + b + c + d;
   uint quotient = sum >> 2;
   uint remainder = sum & 3u;
   return (uint8_t)(quotient +
                    (remainder > 2u ||
                     (remainder == 2u && quotient < 128u)));
}

/*
 * Exact full-level 2x downsample for ordinary four-channel UNORM8 images.
 * RGBA and BGRA are both channel-order agnostic here: each stored byte is
 * filtered independently and written back in the same order.
 */
KERNEL(32)
libagx_mipmap_unorm8x4(global uint8_t *dest, global uint8_t *src,
                       uint dest_width_px, uint dest_height_px,
                       uint src_scale_x, uint src_scale_y,
                       uint dest_tiles_per_row, uint src_tiles_per_row,
                       uint dest_tile_width_log2, uint dest_tile_height_log2,
                       uint src_tile_width_log2, uint src_tile_height_log2)
{
   uint x = cl_global_id.x;
   uint y = cl_global_id.y;
   if (x >= dest_width_px || y >= dest_height_px)
      return;

   uint dest_offset = libagx_copy_image_offset_pow2(
      x, y, 0, dest_tiles_per_row, dest_tile_width_log2,
      dest_tile_height_log2, 4, 1);
   uint src_x = x * src_scale_x;
   uint src_y = y * src_scale_y;
   uint src00 = libagx_copy_image_offset_pow2(
      src_x, src_y, 0, src_tiles_per_row, src_tile_width_log2,
      src_tile_height_log2, 4, 1);
   uint src10 = libagx_copy_image_offset_pow2(
      src_x + src_scale_x - 1, src_y, 0, src_tiles_per_row,
      src_tile_width_log2, src_tile_height_log2, 4, 1);
   uint src01 = libagx_copy_image_offset_pow2(
      src_x, src_y + src_scale_y - 1, 0, src_tiles_per_row,
      src_tile_width_log2, src_tile_height_log2, 4, 1);
   uint src11 = libagx_copy_image_offset_pow2(
      src_x + src_scale_x - 1, src_y + src_scale_y - 1, 0,
      src_tiles_per_row, src_tile_width_log2, src_tile_height_log2, 4, 1);

   for (uint channel = 0; channel < 4; ++channel) {
      dest[dest_offset + channel] = libagx_average4_unorm8(
         src[src00 + channel], src[src10 + channel],
         src[src01 + channel], src[src11 + channel]);
   }
}

/*
 * Exact odd-size full-level downsample for four-channel UNORM8 images.  The
 * CPU builds one packed Q8 coordinate entry per destination X and Y, avoiding
 * every per-pixel divide and all 64-bit arithmetic here.  G17P's sampler
 * rounds the four-tap result half-up to four fractional colour bits; its
 * normalized write rounds exact ties toward the midpoint code 128.
 */
KERNEL(32)
libagx_mipmap_unorm8x4_odd(global uint8_t *dest, global uint8_t *src,
                           global uint *axis,
                           uint dest_width_px, uint dest_height_px,
                           uint dest_tiles_per_row, uint src_tiles_per_row,
                           uint dest_tile_width_log2,
                           uint dest_tile_height_log2,
                           uint src_tile_width_log2,
                           uint src_tile_height_log2)
{
   uint x = cl_global_id.x;
   uint y = cl_global_id.y;
   if (x >= dest_width_px || y >= dest_height_px)
      return;

   uint x_entry = axis[x];
   uint y_entry = axis[dest_width_px + y];
   uint src_x0 = x_entry & 0xffffu;
   uint src_y0 = y_entry & 0xffffu;
   uint wx = (x_entry >> 16) & 0xffu;
   uint wy = (y_entry >> 16) & 0xffu;
   uint src_x1 = src_x0 + (wx != 0u);
   uint src_y1 = src_y0 + (wy != 0u);

   uint dest_offset = libagx_copy_image_offset_pow2(
      x, y, 0, dest_tiles_per_row, dest_tile_width_log2,
      dest_tile_height_log2, 4, 1);
   uint src00 = libagx_copy_image_offset_pow2(
      src_x0, src_y0, 0, src_tiles_per_row, src_tile_width_log2,
      src_tile_height_log2, 4, 1);
   uint src10 = libagx_copy_image_offset_pow2(
      src_x1, src_y0, 0, src_tiles_per_row, src_tile_width_log2,
      src_tile_height_log2, 4, 1);
   uint src01 = libagx_copy_image_offset_pow2(
      src_x0, src_y1, 0, src_tiles_per_row, src_tile_width_log2,
      src_tile_height_log2, 4, 1);
   uint src11 = libagx_copy_image_offset_pow2(
      src_x1, src_y1, 0, src_tiles_per_row, src_tile_width_log2,
      src_tile_height_log2, 4, 1);
   uint ix = 256u - wx;
   uint iy = 256u - wy;

   for (uint channel = 0; channel < 4; ++channel) {
      uint value = src[src00 + channel] * ix * iy +
                   src[src10 + channel] * wx * iy +
                   src[src01 + channel] * ix * wy +
                   src[src11 + channel] * wx * wy;
      uint fixed4 = (value + 2048u) >> 12;
      uint out = fixed4 >> 4;
      uint remainder = fixed4 & 15u;
      out += remainder > 8u || (remainder == 8u && out < 128u);
      dest[dest_offset + channel] = (uint8_t)out;
   }
}

/*
 * Resolve an unorm8x4 multisample surface into a single-sample surface.
 * AIL tiles one logical pixel as a block containing every sample. Averaging
 * bytes is format-order agnostic for the four 8-bit UNORM channel formats
 * this entry point accepts on the CPU side.
 */
KERNEL(32)
libagx_resolve_unorm8x4(global uint8_t *dest, global uint8_t *src,
                        uint width_px, uint height_px,
                        uint dest_x_px, uint dest_y_px,
                        uint src_x_px, uint src_y_px,
                        uint dest_pitch_B, uint dest_stride_el,
                        uint src_stride_el,
                        uint dest_tile_width_el, uint dest_tile_height_el,
                        uint src_tile_width_el, uint src_tile_height_el,
                        uint sample_count, uint dest_tiled, uint src_tiled,
                        uint resolve_mode)
{
   uint id = cl_global_id.x;
   if (id >= width_px * height_px)
      return;

   uint x = id % width_px;
   uint y = id / width_px;
   uint dest_offset_B = libagx_copy_image_offset(
      dest_x_px + x, dest_y_px + y, dest_pitch_B, dest_stride_el,
      dest_tile_width_el, dest_tile_height_el, 4, dest_tiled);

   for (uint b = 0; b < 4; ++b) {
      uint sum = 0;
      uint extrema = resolve_mode == 1 ? 255 : 0;
      for (uint s = 0; s < sample_count; ++s) {
         uint src_offset_B = libagx_copy_image_offset(
            src_x_px + x, src_y_px + y, 0, src_stride_el,
            src_tile_width_el, src_tile_height_el,
            4 * sample_count, src_tiled);
         uint value = src[src_offset_B + s * 4 + b];
         sum += value;
         extrema = resolve_mode == 1 ? min(extrema, value)
                                      : max(extrema, value);
      }

      dest[dest_offset_B + b] = resolve_mode
                                   ? extrema
                                   : (sum + sample_count / 2) / sample_count;
   }
}

KERNEL(32)
libagx_fill_uint4(global uint4 *address, uint layer_stride_uint4, uint a,
                  uint b, uint c, uint d)
{
   uint offs = cl_global_id.x + cl_global_id.y * layer_stride_uint4;
   address[offs] = (uint4)(a, b, c, d);
}
