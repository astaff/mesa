/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_memory_budget.h"
#include "vk_alloc.h"
#include "wsi_common_private.h"
#include "wsi_common_entrypoints.h"
#include "drm-uapi/drm_fourcc.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#define MIB (UINT64_C(1024) * 1024)
static const uint64_t limit = 2048 * MIB, host = 384 * MIB;
static uint64_t pinned, cached, peak;
static unsigned bo_calls, kernel_oom_attempts, trims;
static unsigned passing, host_live;
static bool bind_fails, bo_fails, image_bind_fails, export_fails, limited = true;
static unsigned images_live, successes, exported_count;
static int exported_fds[4];

__typeof__(agx_bo_create) __wrap_agx_bo_create;
__typeof__(agx_bo_unreference) __wrap_agx_bo_unreference;
__typeof__(agx_bo_bind) __wrap_agx_bo_bind;
__typeof__(agx_bo_cache_evict_all) __wrap_agx_bo_cache_evict_all;
__typeof__(hk_get_cgroup_memory_available) __wrap_hk_get_cgroup_memory_available;
__typeof__(__vk_errorf) __wrap___vk_errorf;

/* These replace only the kernel-facing allocation operations. The test calls
 * the production Vulkan allocation and free entrypoints and object allocator. */
struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
                     enum agx_bo_flags flags, const char *label)
{
   ++bo_calls;
   if (limited && size > limit - host - pinned - cached) {
      ++kernel_oom_attempts;
      return NULL;
   }
   if (bo_fails)
      return NULL;
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   assert(bo);
   bo->size = size;
   bo->flags = flags;
   bo->va = calloc(1, sizeof(*bo->va));
   assert(bo->va);
   bo->va->addr = UINT64_C(0x1000000000) + pinned;
   pinned += size;
   if (pinned > peak)
      peak = pinned;
   return bo;
}

void
__wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo)
      return;
   assert(pinned >= bo->size);
   pinned -= bo->size;
   free(bo->va);
   free(bo);
}

int
__wrap_agx_bo_bind(struct agx_device *dev, struct agx_bo *bo, uint64_t address,
                   uint64_t size, uint64_t offset, uint32_t flags)
{
   return bind_fails ? -1 : 0;
}

void
__wrap_agx_bo_cache_evict_all(struct agx_device *dev)
{
   ++trims;
   cached = 0;
}

bool
__wrap_hk_get_cgroup_memory_available(const char *cgroup, const char *mountinfo,
                                     uint64_t *available)
{
   *available = limit - host - pinned - cached;
   return limited;
}

VkResult
__wrap___vk_errorf(const void *object, VkResult result,
                   const char *file, int line, const char *format, ...)
{
   return result;
}

static void * VKAPI_PTR
allocate(void *data, size_t size, size_t alignment, VkSystemAllocationScope scope)
{
   if (!passing)
      return NULL;
   --passing;
   void *ptr = vk_default_allocator()->pfnAllocation(NULL, size, alignment, scope);
   if (ptr)
      ++host_live;
   return ptr;
}

static void VKAPI_PTR
release(void *data, void *ptr)
{
   if (ptr) {
      assert(host_live);
      --host_live;
      vk_default_allocator()->pfnFree(NULL, ptr);
   }
}

static VkAllocationCallbacks allocator;

struct test_image {
   uint64_t size;
};

static VkResult VKAPI_PTR
create_image(VkDevice device, const VkImageCreateInfo *info,
             const VkAllocationCallbacks *callbacks, VkImage *image)
{
   struct test_image *test = vk_alloc(callbacks, sizeof(*test), 8,
                                      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!test)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   test->size = (uint64_t)info->extent.width * info->extent.height * 4;
   *image = (VkImage)(uintptr_t)test;
   ++images_live;
   return VK_SUCCESS;
}

static void VKAPI_PTR
destroy_image(VkDevice device, VkImage image,
              const VkAllocationCallbacks *callbacks)
{
   if (image) {
      assert(images_live);
      --images_live;
      vk_free(callbacks, (void *)(uintptr_t)image);
   }
}

static void VKAPI_PTR
image_requirements(VkDevice device, VkImage image, VkMemoryRequirements *req)
{
   const struct test_image *test = (void *)(uintptr_t)image;
   *req = (VkMemoryRequirements){
      .size = test->size, .alignment = 16384, .memoryTypeBits = 1,
   };
}

static VkResult VKAPI_PTR
bind_image(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize offset)
{
   return image_bind_fails ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
}

static VkResult VKAPI_PTR
memory_fd(VkDevice device, const VkMemoryGetFdInfoKHR *info, int *fd)
{
   *fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
   assert(*fd >= 0);
   assert(exported_count < 4);
   exported_fds[exported_count++] = *fd;
   return export_fails ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
}

static void VKAPI_PTR
format_properties(VkPhysicalDevice physical, VkFormat format, VkFormatProperties2 *props)
{
   VkDrmFormatModifierPropertiesListEXT *list = props->pNext;
   list->drmFormatModifierCount = 1;
   if (list->pDrmFormatModifierProperties) {
      list->pDrmFormatModifierProperties[0] = (VkDrmFormatModifierPropertiesEXT){
         .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
         .drmFormatModifierPlaneCount = 1,
      };
   }
}

static VkResult VKAPI_PTR
image_format_properties(VkPhysicalDevice physical,
                        const VkPhysicalDeviceImageFormatInfo2 *info,
                        VkImageFormatProperties2 *props)
{
   props->imageFormatProperties.maxExtent = (VkExtent3D){16384, 16384, 1};
   return VK_SUCCESS;
}

static VkResult VKAPI_PTR
image_modifier(VkDevice device, VkImage image, VkImageDrmFormatModifierPropertiesEXT *props)
{
   props->drmFormatModifier = DRM_FORMAT_MOD_LINEAR;
   return VK_SUCCESS;
}

static void VKAPI_PTR
image_layout(VkDevice device, VkImage image, const VkImageSubresource *subresource,
             VkSubresourceLayout *layout)
{
   const struct test_image *test = (void *)(uintptr_t)image;
   *layout = (VkSubresourceLayout){.size = test->size};
}

#define NULL_DESTROY(name, type) \
static void VKAPI_PTR name(VkDevice device, type handle, \
                           const VkAllocationCallbacks *callbacks) \
{ assert(handle == VK_NULL_HANDLE); }
NULL_DESTROY(destroy_query, VkQueryPool)
NULL_DESTROY(destroy_buffer, VkBuffer)
NULL_DESTROY(destroy_semaphore, VkSemaphore)
NULL_DESTROY(destroy_fence, VkFence)

static void
init_wsi(struct hk_device *dev)
{
   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct wsi_device *wsi = &pdev->wsi_device;
   pdev->vk.wsi_device = wsi;
   *wsi = (struct wsi_device){
      .pdevice = hk_physical_device_to_handle(pdev),
      .supports_modifiers = true,
      .override_present_mode = VK_PRESENT_MODE_MAX_ENUM_KHR,
      .CreateImage = create_image, .DestroyImage = destroy_image,
      .GetImageMemoryRequirements = image_requirements,
      .AllocateMemory = hk_AllocateMemory, .FreeMemory = hk_FreeMemory,
      .BindImageMemory = bind_image, .GetMemoryFdKHR = memory_fd,
      .GetPhysicalDeviceFormatProperties2 = format_properties,
      .GetPhysicalDeviceImageFormatProperties2 = image_format_properties,
      .GetImageDrmFormatModifierPropertiesEXT = image_modifier,
      .GetImageSubresourceLayout = image_layout,
      .DestroyQueryPool = destroy_query, .DestroyBuffer = destroy_buffer,
      .DestroySemaphore = destroy_semaphore, .DestroyFence = destroy_fence,
   };
   wsi->memory_props.memoryTypeCount = 1;
   wsi->memory_props.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
   assert(wsi_headless_init_wsi(wsi, vk_default_allocator(), wsi->pdevice) == VK_SUCCESS);
}

static VkResult
alloc(struct hk_device *dev, uint64_t size, VkDeviceMemory *memory)
{
   const VkMemoryAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = size,
      .memoryTypeIndex = 0,
   };
   return hk_AllocateMemory(hk_device_to_handle(dev), &info, &allocator, memory);
}

static void
swapchain_attempt(struct hk_device *dev, uint64_t image_size)
{
   exported_count = 0;
   VkIcdSurfaceHeadless surface = {.base.platform = VK_ICD_WSI_PLATFORM_HEADLESS};
   const VkSwapchainCreateInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = (VkSurfaceKHR)(uintptr_t)&surface,
      .minImageCount = 4,
      .imageFormat = VK_FORMAT_R8G8B8A8_UNORM,
      .imageExtent = {image_size == 1024 * MIB ? 16384 : 8192,
                     image_size == 1024 * MIB ? 16384 : 8192},
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .presentMode = VK_PRESENT_MODE_FIFO_KHR,
   };
   VkSwapchainKHR swapchain = VK_NULL_HANDLE;
   VkResult result = wsi_CreateSwapchainKHR(hk_device_to_handle(dev), &info,
                                            &allocator, &swapchain);
   if (image_size == 1024 * MIB)
      assert(result != VK_SUCCESS);
   assert(result == VK_SUCCESS || result == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
          result == VK_ERROR_OUT_OF_HOST_MEMORY);
   if (result == VK_SUCCESS) {
      ++successes;
      wsi_DestroySwapchainKHR(hk_device_to_handle(dev), swapchain, &allocator);
   }
   assert(!pinned && !host_live && !images_live);
   for (unsigned i = 0; i < exported_count; ++i)
      assert(fcntl(exported_fds[i], F_GETFD) == -1 && errno == EBADF);
   assert(!hk_device_physical(dev)->mem_heaps[0].used);
   assert(!kernel_oom_attempts);
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   assert(dev && pdev);
   dev->vk.alloc = *vk_default_allocator();
   allocator = dev->vk.alloc;
   allocator.pfnAllocation = allocate;
   allocator.pfnFree = release;
   dev->vk.physical = &pdev->vk;
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);
   vk_object_base_init(NULL, &pdev->vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   pdev->mem_types[0].heapIndex = 0;
   pdev->mem_heaps[0].size = 8 * 1024 * MIB;
   simple_mtx_init(&dev->dev.bo_map_lock, mtx_plain);
   init_wsi(dev);

   /* The CTS tries callback limits 0..1024, including an extent requiring
    * four 1 GiB images. No attempt may enter the kernel above the hard limit,
    * and every completed image must be freed on the next image's failure. */
   for (unsigned count = 0; count <= 1024; ++count) {
      passing = count;
      swapchain_attempt(dev, 1024 * MIB);
      passing = count;
      swapchain_attempt(dev, 256 * MIB);
   }
   assert(peak == 1024 * MIB);
   assert(trims && bo_calls && successes);

   /* Preserve the current error paths after backing was allocated. */
   passing = 1024;
   bind_fails = true;
   unsigned before = bo_calls;
   swapchain_attempt(dev, 256 * MIB);
   assert(bo_calls == before + 1);
   bind_fails = false;
   bo_fails = true;
   passing = 1024;
   before = bo_calls;
   swapchain_attempt(dev, 256 * MIB);
   assert(bo_calls == before + 1);
   bo_fails = false;
   passing = 1024;
   image_bind_fails = true;
   swapchain_attempt(dev, 256 * MIB);
   assert(exported_count == 1);
   image_bind_fails = false;
   passing = 1024;
   export_fails = true;
   swapchain_attempt(dev, 256 * MIB);
   assert(exported_count == 1);
   export_fails = false;

   /* Cached objects are reclaimed and admission retried before returning OOM. */
   cached = 1024 * MIB;
   passing = 4;
   VkDeviceMemory memory = 0;
   before = trims;
   assert(alloc(dev, 1024 * MIB, &memory) == VK_SUCCESS);
   assert(trims == before + 1 && !cached);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   assert(!pinned && !host_live);

   /* Physical heap and alignment failures must also precede BO allocation. */
   pdev->mem_heaps[0].size = 512 * MIB;
   before = bo_calls;
   assert(alloc(dev, 1024 * MIB, &memory) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(alloc(dev, UINT64_MAX, &memory) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(bo_calls == before);
   pdev->mem_heaps[0].size = 8 * 1024 * MIB;

   /* Missing/unlimited cgroups keep normal allocation behavior. */
   limited = false;
   passing = 4;
   assert(alloc(dev, 1024 * MIB, &memory) == VK_SUCCESS);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   assert(!pinned && !host_live && !kernel_oom_attempts);
   wsi_headless_finish_wsi(&pdev->wsi_device, vk_default_allocator());
   vk_object_base_finish(&dev->vk.base);
   vk_object_base_finish(&pdev->vk.base);
   simple_mtx_destroy(&dev->dev.bo_map_lock);
   free(dev);
   free(pdev);
   puts("PASS: 2050 swapchain allocation attempts, no kernel OOM entry or retained backing");
   return 0;
}
