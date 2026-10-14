#pragma once

#include "compiler/nir/nir.h"

static inline bool
agx_aligned_bound_constant(nir_scalar scalar, uint32_t value)
{
   scalar = nir_scalar_chase_movs(scalar);
   return nir_scalar_is_const(scalar) &&
          nir_scalar_as_uint(scalar) == value;
}

static inline bool
agx_aligned_bound_add(nir_scalar value, nir_scalar offset, uint32_t delta)
{
   value = nir_scalar_chase_movs(value);
   offset = nir_scalar_chase_movs(offset);
   if (delta == 0)
      return nir_scalar_equal(value, offset);
   if (!nir_scalar_is_alu(value) || nir_scalar_alu_op(value) != nir_op_iadd)
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar src = nir_scalar_chase_alu_src(value, i);
      if (nir_scalar_equal(nir_scalar_chase_movs(src), offset) &&
          agx_aligned_bound_constant(nir_scalar_chase_alu_src(value, 1 - i),
                                     delta))
         return true;
   }
   return false;
}

static bool
agx_has_reusable_bound_store_index(const nir_alu_instr *instr)
{
   if (instr->op != nir_op_ult || instr->def.num_components != 1)
      return false;
   nir_scalar end = nir_scalar_chase_movs(
      nir_get_scalar(instr->src[0].src.ssa, instr->src[0].swizzle[0]));
   if (!nir_scalar_is_alu(end) || nir_scalar_alu_op(end) != nir_op_iadd)
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar masked = nir_scalar_chase_alu_src(end, i);
      nir_scalar constant = nir_scalar_chase_alu_src(end, 1 - i);
      if (!nir_scalar_is_alu(masked) ||
          nir_scalar_alu_op(masked) != nir_op_iand ||
          !nir_scalar_is_const(constant))
         continue;
      for (unsigned j = 0; j < 2; ++j) {
         nir_scalar offset = nir_scalar_chase_alu_src(masked, j);
         nir_scalar mask = nir_scalar_chase_alu_src(masked, 1 - j);
         if (!nir_scalar_is_const(mask))
            continue;
         uint32_t size = -(uint32_t)nir_scalar_as_uint(mask);
         if (size != 2 && size != 4)
            continue;
         uint32_t end_delta = nir_scalar_as_uint(constant);
         if ((end_delta & (size - 1)) != size - 1)
            continue;
         uint32_t delta = end_delta - (size - 1);
         unsigned shift = size == 2 ? 1 : 2;
         nir_foreach_if_use(use, &instr->def) {
            nir_block *block = nir_if_first_then_block(nir_src_use_if(use));
            nir_foreach_instr(store_instr, block) {
               if (store_instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *store = nir_instr_as_intrinsic(store_instr);
               if (store->intrinsic != nir_intrinsic_store_agx ||
                   nir_src_bit_size(store->src[0]) != size * 8 ||
                   store->num_components != 1 ||
                   nir_intrinsic_base(store) != 0 ||
                   nir_intrinsic_sign_extend(store))
                  continue;
               nir_scalar index = nir_scalar_chase_movs(
                  nir_get_scalar(store->src[2].ssa, 0));
               if (!nir_scalar_is_alu(index) ||
                   nir_scalar_alu_op(index) != nir_op_ushr ||
                   !agx_aligned_bound_constant(
                      nir_scalar_chase_alu_src(index, 1), shift))
                  continue;
               if (agx_aligned_bound_add(nir_scalar_chase_alu_src(index, 0),
                                        offset, delta))
                  return true;
            }
         }
      }
   }
   return false;
}
