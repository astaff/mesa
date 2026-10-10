/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_g17p_grid_helper_core.h"

#include <string.h>

static const char agx3_g17p_grid_helper_msl[] =
   "#include <metal_stdlib>\n"
   "using namespace metal;\n"
   "\n"
   "kernel void g17p_grid_setup(device const uint *threadgroups [[buffer(0)]],\n"
   "                            device const uint *threads_per_group [[buffer(1)]],\n"
   "                            device uint *geometry [[buffer(2)]],\n"
   "                            uint index [[thread_position_in_grid]])\n"
   "{\n"
   "    if (index != 0)\n"
   "        return;\n"
   "\n"
   "    geometry[0] = threadgroups[0] * threads_per_group[0];\n"
   "    geometry[1] = threadgroups[1] * threads_per_group[1];\n"
   "    geometry[2] = threadgroups[2] * threads_per_group[2];\n"
   "    geometry[3] = threads_per_group[0];\n"
   "    geometry[4] = threads_per_group[1];\n"
   "    geometry[5] = threads_per_group[2];\n"
   "}\n";

const char *
agx3_g17p_grid_helper_source(void)
{
   return agx3_g17p_grid_helper_msl;
}
