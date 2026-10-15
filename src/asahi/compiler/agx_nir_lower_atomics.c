/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"

static bool
needs_cas_loop(const nir_intrinsic_instr *intr, bool native_fadd)
{
   if (intr->intrinsic != nir_intrinsic_global_atomic &&
       intr->intrinsic != nir_intrinsic_shared_atomic)
      return false;

   switch (nir_intrinsic_atomic_op(intr)) {
   case nir_atomic_op_fadd:
      return intr->def.bit_size != 32 || !native_fadd ||
             intr->intrinsic == nir_intrinsic_shared_atomic;
   case nir_atomic_op_fmin:
   case nir_atomic_op_fmax:
      return true;
   default:
      return false;
   }
}

static nir_def *
load_current(nir_builder *b, nir_intrinsic_instr *intr)
{
   const unsigned bit_size = intr->def.bit_size;
   nir_def *addr = intr->src[0].ssa;

   if (intr->intrinsic == nir_intrinsic_shared_atomic) {
      return nir_load_shared(b, 1, bit_size, addr,
                             .base = nir_intrinsic_base(intr),
                             .align_mul = bit_size / 8,
                             .access = nir_intrinsic_access(intr) |
                                       ACCESS_ATOMIC);
   }

   return nir_load_global(b, 1, bit_size, addr,
                          .align_mul = bit_size / 8,
                          .access = nir_intrinsic_access(intr) |
                                    ACCESS_ATOMIC | ACCESS_COHERENT);
}

static nir_def *
compare_swap(nir_builder *b, nir_intrinsic_instr *intr, nir_def *compare,
             nir_def *replacement)
{
   const unsigned bit_size = intr->def.bit_size;
   nir_def *addr = intr->src[0].ssa;

   if (intr->intrinsic == nir_intrinsic_shared_atomic) {
      return nir_shared_atomic_swap(b, bit_size, addr, compare, replacement,
                                    .base = nir_intrinsic_base(intr),
                                    .atomic_op = nir_atomic_op_cmpxchg,
                                    .access = nir_intrinsic_access(intr));
   }

   return nir_global_atomic_swap(b, bit_size, addr, compare, replacement,
                                 .atomic_op = nir_atomic_op_cmpxchg,
                                 .access = nir_intrinsic_access(intr));
}

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (!needs_cas_loop(intr, *(bool *)data))
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   const nir_op alu_op = nir_atomic_op_to_alu(nir_intrinsic_atomic_op(intr));
   nir_def *operand = intr->src[1].ssa;
   nir_def *initial = load_current(b, intr);

   nir_phi_instr *phi = nir_phi_instr_create(b->shader);
   nir_def_init(&phi->instr, &phi->def, 1, intr->def.bit_size);

   nir_loop *loop = nir_push_loop(b);
   nir_phi_instr_add_src(phi, nir_def_block(initial), initial);
   nir_def *old = &phi->def;
   nir_def *updated = nir_build_alu2(b, alu_op, old, operand);
   nir_def *seen = compare_swap(b, intr, old, updated);
   nir_break_if(b, nir_ieq(b, seen, old));
   nir_phi_instr_add_src(phi, nir_loop_last_block(loop), seen);
   nir_pop_loop(b, loop);

   b->cursor = nir_before_block(nir_loop_first_block(loop));
   nir_builder_instr_insert(b, &phi->instr);

   nir_def_replace(&intr->def, seen);
   return true;
}

bool
agx_nir_lower_atomics(nir_shader *s, bool native_fadd)
{
   return nir_shader_intrinsics_pass(s, lower, nir_metadata_none,
                                     &native_fadd);
}
