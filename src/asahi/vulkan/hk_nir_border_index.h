#pragma once
#include "compiler/nir/nir.h"

static inline bool
hk_nir_border_address(nir_scalar value, nir_scalar *base, uint64_t *offset)
{
   value = nir_scalar_chase_movs(value);
   nir_intrinsic_instr *load = nir_scalar_as_intrinsic(value);
   if (!load || value.comp || load->intrinsic != nir_intrinsic_load_global_constant ||
       load->def.bit_size != 16 || load->def.num_components != 1 ||
       (nir_intrinsic_access(load) & ACCESS_VOLATILE))
      return false;
   nir_scalar addr = nir_scalar_chase_movs(nir_get_scalar(load->src[0].ssa, 0));
   if (addr.def->bit_size != 64 || !nir_scalar_is_alu(addr) ||
       nir_scalar_alu_op(addr) != nir_op_iadd)
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar constant = nir_scalar_chase_movs(nir_scalar_chase_alu_src(addr, i));
      if (nir_scalar_is_const(constant)) {
         *offset = nir_scalar_as_uint(constant);
         *base = nir_scalar_chase_movs(nir_scalar_chase_alu_src(addr, i ^ 1));
         return true;
      }
   }
   return false;
}

static inline bool
hk_nir_border_indices_same_descriptor(nir_scalar original, nir_scalar unbiased,
                                      uint64_t delta)
{
   nir_scalar original_base, unbiased_base;
   uint64_t original_offset, unbiased_offset;
   return hk_nir_border_address(original, &original_base, &original_offset) &&
          hk_nir_border_address(unbiased, &unbiased_base, &unbiased_offset) &&
          nir_scalar_equal(original_base, unbiased_base) &&
          original_offset - unbiased_offset == delta;
}
