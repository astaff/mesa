/* SPDX-License-Identifier: MIT */
#include "compiler/nir/nir_builder.h"
#include "util/hash_table.h"
#include "agx_nir.h"

struct scalar32 {
   nir_scalar value;
   uint32_t constant;
   uint32_t bound;
   bool is_constant;
   bool is_boolean;
};

static bool
zero_extended32(nir_shader *shader, struct hash_table *ranges, nir_scalar value,
                struct scalar32 *low)
{
   value = nir_scalar_chase_movs(value);
   if (value.def->bit_size != 64)
      return false;
   if (nir_scalar_is_const(value)) {
      const uint64_t constant = nir_scalar_as_uint(value);
      if (constant > UINT32_MAX)
         return false;
      *low = (struct scalar32){.constant = constant,
                               .bound = constant,
                               .is_constant = true};
      return true;
   }
   if (!nir_scalar_is_alu(value))
      return false;

   const nir_op op = nir_scalar_alu_op(value);
   if (op == nir_op_pack_64_2x32_split) {
      const nir_scalar high = nir_scalar_chase_alu_src(value, 1);
      if (!nir_scalar_is_const(high) || nir_scalar_as_uint(high))
         return false;
   } else if (op != nir_op_u2u64 && op != nir_op_i2i64 && op != nir_op_b2i64) {
      return false;
   }
   const nir_scalar src = nir_scalar_chase_alu_src(value, 0);
   if (src.def->bit_size > 32)
      return false;
   const uint32_t bound = nir_unsigned_upper_bound(shader, ranges, src);
   if (op == nir_op_i2i64 &&
       bound > ((UINT64_C(1) << (src.def->bit_size - 1)) - 1))
      return false;
   *low = (struct scalar32){.value = src,
                            .bound = bound,
                            .is_boolean = op == nir_op_b2i64};
   return true;
}

static nir_def *
get_low32(nir_builder *b, const struct scalar32 *low)
{
   if (low->is_constant)
      return nir_imm_int(b, low->constant);
   nir_def *value = nir_channel(b, low->value.def, low->value.comp);
   return low->is_boolean ? nir_b2i32(b, value) : nir_u2u32(b, value);
}

static bool
scalarize(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->def.bit_size != 64 || alu->def.num_components != 1)
      return false;
   const bool right = alu->op == nir_op_ushr || alu->op == nir_op_ishr;
   const bool left = alu->op == nir_op_ishl;
   const bool shift = right || left;
   switch (alu->op) {
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
   case nir_op_ushr:
   case nir_op_ishr:
   case nir_op_ishl:
      break;
   default:
      return false;
   }

   struct hash_table *ranges = data;
   nir_scalar src[2] = {
      nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), 0),
      nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), 1)};
   struct scalar32 low[2] = {0};
   bool narrow[2] = {zero_extended32(b->shader, ranges, src[0], &low[0]),
                     false};
   uint32_t count_bound = UINT32_MAX;
   if (shift) {
      if (!narrow[0] || src[1].def->bit_size != 32)
         return false;
      count_bound = nir_unsigned_upper_bound(b->shader, ranges, src[1]);
      if (left &&
          (count_bound >= 32 || low[0].bound > (UINT32_MAX >> count_bound)))
         return false;
   } else {
      narrow[1] = zero_extended32(b->shader, ranges, src[1], &low[1]);
      if (alu->op == nir_op_iand) {
         if (!narrow[0] && !narrow[1])
            return false;
      } else if (!narrow[0] || !narrow[1]) {
         return false;
      }
   }

   b->cursor = nir_before_instr(instr);
   nir_def *a = narrow[0]
                   ? get_low32(b, &low[0])
                   : nir_u2u32(b, nir_channel(b, src[0].def, src[0].comp));
   nir_def *value;
   if (shift) {
      nir_def *count = nir_channel(b, src[1].def, src[1].comp);
      if (right && count_bound >= 32)
         count = nir_iand_imm(b, count, 63);
      value = right ? nir_ushr(b, a, count) : nir_ishl(b, a, count);
      if (right && count_bound >= 32)
         value =
            nir_bcsel(b, nir_ult_imm(b, count, 32), value, nir_imm_int(b, 0));
   } else {
      nir_def *c = narrow[1]
                      ? get_low32(b, &low[1])
                      : nir_u2u32(b, nir_channel(b, src[1].def, src[1].comp));
      value = nir_build_alu(b, alu->op, a, c, NULL, NULL);
   }
   nir_def_rewrite_uses(&alu->def, nir_u2u64(b, value));
   nir_instr_remove(instr);
   return true;
}

bool
agx_nir_scalarize_int64(nir_shader *shader)
{
   bool progress = false;
   nir_foreach_function_impl(impl, shader) {
      struct hash_table *ranges = _mesa_pointer_hash_table_create(NULL);
      progress |= nir_function_instructions_pass(
         impl, scalarize, nir_metadata_control_flow, ranges);
      _mesa_hash_table_destroy(ranges, NULL);
   }
   return progress;
}
