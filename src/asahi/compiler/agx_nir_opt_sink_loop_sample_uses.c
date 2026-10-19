/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir.h"

/*
 * Sink the ALU consumers of a loop body's texture results to the latest
 * top-level block of that body that still dominates every use.
 *
 * Mesa's order consumes each texture result right after its sample, so a
 * loop whose body samples several times completes those samples one at a
 * time: twelve serial texture round trips per iteration in the robustness2
 * sampled_image 3D vertex loop. Apple's compiler for the same straight-line
 * loop (native G17G witness) issues all twelve
 * SMP in source order, the three conditional ones inside their own
 * cndst/cndend, and places every compare and accumulate of the twelve
 * results after the last SMP, so the iteration waits for the samples once.
 *
 * nir_opt_sink leaves general ALU alone and nir_opt_move only reorders
 * within a block; the backend pressure and ILP schedules are per block too,
 * and a sample under a condition (c != 0 here) splits the body into several
 * blocks. Moving a pure ALU instruction to a block its definition dominates
 * and that dominates all of its uses changes neither its value nor the SSA
 * dominance of the program; it may only run less often (it no longer runs on
 * paths that leave the body early). Texture instructions themselves never
 * move, so no sample is speculated.
 *
 * The sunk results stay live until the end of the body, so the flagged
 * sample results are bounded per loop by AGX_NIR_LOOP_SAMPLE_SINK_HALVES
 * (twelve 32-bit vec4 results, the witness loop); consumers of later samples
 * keep their place.
 */

#define AGX_NIR_LOOP_SAMPLE_SINK_HALVES 96

/* pass_flags bit: the instruction is, or depends on, a sample of this loop
 * whose consumers may sink. */
#define SAMPLE_DEPENDENT 1

static unsigned
def_halves(const nir_def *def)
{
   return def->num_components * DIV_ROUND_UP(def->bit_size, 16);
}

/* nir_foreach_src callback: keep going while the source is independent, so
 * the walk returns false exactly when some source is sample dependent. */
static bool
src_independent(nir_src *src, void *data)
{
   return (nir_def_instr(src->ssa)->pass_flags & SAMPLE_DEPENDENT) == 0;
}

/* The top-level block of the loop body containing (or preceding the nested
 * control flow containing) block b, or NULL when b is outside the loop. */
static nir_block *
loop_top_level_block(nir_loop *loop, nir_block *b)
{
   nir_cf_node *node = &b->cf_node;
   while (node->parent != &loop->cf_node) {
      node = node->parent;
      if (!node)
         return NULL;
   }
   if (node->type == nir_cf_node_block)
      return nir_cf_node_as_block(node);
   /* Nested if or loop: the top-level block right before it. */
   nir_cf_node *prev = nir_cf_node_prev(node);
   assert(prev && prev->type == nir_cf_node_block);
   return nir_cf_node_as_block(prev);
}

static bool
sink_loop(nir_loop *loop)
{
   if (nir_loop_has_continue_construct(loop))
      return false;

   /* Mark the body's samples, within the result budget, and everything
    * computed from them. Loop-carried values reach their consumers through
    * header phis, so iterate to a fixed point. */
   unsigned budget = AGX_NIR_LOOP_SAMPLE_SINK_HALVES;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block)
         instr->pass_flags = 0;
   }
   bool any_sample = false;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_tex)
            continue;
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         const unsigned halves = def_halves(&tex->def);
         if (halves > budget)
            continue;
         budget -= halves;
         instr->pass_flags = SAMPLE_DEPENDENT;
         any_sample = true;
      }
   }
   if (!any_sample)
      return false;

   bool changed;
   do {
      changed = false;
      nir_foreach_block_in_cf_node(block, &loop->cf_node) {
         nir_foreach_instr(instr, block) {
            if (instr->pass_flags & SAMPLE_DEPENDENT)
               continue;
            if (instr->type != nir_instr_type_alu &&
                instr->type != nir_instr_type_phi)
               continue;
            if (!nir_foreach_src(instr, src_independent, NULL)) {
               instr->pass_flags = SAMPLE_DEPENDENT;
               changed = true;
            }
         }
      }
   } while (changed);

   /* Consumers first: visiting in reverse lets a producer follow the uses
    * that already moved. */
   bool progress = false;
   nir_foreach_block_in_cf_node_reverse_safe(block, &loop->cf_node) {
      if (block->cf_node.parent != &loop->cf_node)
         continue;
      nir_foreach_instr_reverse_safe(instr, block) {
         if (instr->type != nir_instr_type_alu ||
             !(instr->pass_flags & SAMPLE_DEPENDENT))
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);

         nir_block *lca = NULL;
         nir_foreach_use_including_if(use, &alu->def)
            lca = nir_dominance_lca(lca, nir_src_get_block(use));
         if (!lca)
            continue;

         nir_block *target = loop_top_level_block(loop, lca);
         if (!target || target->index <= block->index)
            continue;
         assert(nir_block_dominates(block, target));

         nir_instr_remove(instr);
         nir_instr_insert(nir_after_phis(target), instr);
         progress = true;
      }
   }
   return progress;
}

static bool
sink_cf_list(struct exec_list *list)
{
   bool progress = false;
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= sink_cf_list(&nif->then_list);
         progress |= sink_cf_list(&nif->else_list);
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         progress |= sink_cf_list(&loop->body);
         progress |= sink_loop(loop);
      }
   }
   return progress;
}

bool
agx_nir_opt_sink_loop_sample_uses(nir_shader *shader)
{
   bool progress = false;
   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_block_index |
                                    nir_metadata_dominance |
                                    nir_metadata_dominance_lca);
      const bool impl_progress = sink_cf_list(&impl->body);
      progress |= nir_progress(impl_progress, impl, nir_metadata_control_flow);
   }
   return progress;
}
