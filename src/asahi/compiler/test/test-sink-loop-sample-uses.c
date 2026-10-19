/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * agx_nir_opt_sink_loop_sample_uses: a loop body's sample consumers move to
 * the latest top-level block dominating their uses; samples never move;
 * results beyond the per-loop budget keep their consumers in place.
 */
#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/glsl_types.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned failures;

#define CHECK(cond)                                                          \
   do {                                                                      \
      if (!(cond)) {                                                         \
         fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__,   \
                 #cond);                                                     \
         failures++;                                                         \
      }                                                                      \
   } while (0)

static nir_def *
txf3d(nir_builder *b, nir_def *c, unsigned bit_size)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_3D;
   tex->dest_type = bit_size == 16 ? nir_type_int16 : nir_type_int32;
   tex->coord_components = 3;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                     nir_vec3(b, c, nir_imm_int(b, 0),
                                              nir_imm_int(b, 0)));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, bit_size);
   nir_builder_instr_insert(b, &tex->instr);
   return &tex->def;
}

static nir_block *
def_block(nir_def *def)
{
   return nir_def_block(def);
}

struct loop_shader {
   nir_builder b;
   nir_loop *loop;
   nir_phi_instr *c_phi, *acc_phi;
   nir_def *c, *acc;
};

/* for (c = 100, acc = 0;; --c) { body; if (c - 1 < 0) break; } */
static void
loop_begin(struct loop_shader *s)
{
   s->b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, NULL, "loop");
   nir_builder *b = &s->b;
   nir_def *c0 = nir_imm_int(b, 100);
   nir_def *acc0 = nir_imm_int(b, 0);
   s->loop = nir_push_loop(b);
   s->c_phi = nir_phi_instr_create(b->shader);
   s->acc_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&s->c_phi->instr, &s->c_phi->def, 1, 32);
   nir_def_init(&s->acc_phi->instr, &s->acc_phi->def, 1, 32);
   nir_block *preheader = nir_cf_node_as_block(nir_cf_node_prev(&s->loop->cf_node));
   nir_phi_instr_add_src(s->c_phi, preheader, c0);
   nir_phi_instr_add_src(s->acc_phi, preheader, acc0);
   nir_builder_instr_insert(b, &s->c_phi->instr);
   nir_builder_instr_insert(b, &s->acc_phi->instr);
   s->c = &s->c_phi->def;
   s->acc = &s->acc_phi->def;
}

static void
loop_end(struct loop_shader *s, nir_def *acc)
{
   nir_builder *b = &s->b;
   nir_def *next = nir_iadd_imm(b, s->c, -1);
   nir_push_if(b, nir_ilt_imm(b, next, 0));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_block *latch = nir_loop_last_block(s->loop);
   nir_phi_instr_add_src(s->c_phi, latch, next);
   nir_phi_instr_add_src(s->acc_phi, latch, acc);
   nir_pop_loop(b, s->loop);
   nir_store_global(b, acc, nir_imm_int64(b, 0x1000), .write_mask = 1,
                    .align_mul = 4);
   nir_validate_shader(b->shader, "before");
}

/* The witness shape: sample, consume, conditional sample, consume, sample,
 * consume. After the pass every consumer sits in the last top-level block
 * (the one holding the final sample), and no sample moved. */
static void
test_witness_shape(void)
{
   struct loop_shader s;
   loop_begin(&s);
   nir_builder *b = &s.b;

   nir_def *t0 = txf3d(b, s.c, 32);
   nir_def *n0 = nir_ineg(b, nir_channel(b, t0, 0));
   nir_def *a1 = nir_iadd(b, s.acc, n0);

   nir_push_if(b, nir_ine_imm(b, s.c, 0));
   nir_def *t1 = txf3d(b, s.c, 32);
   nir_def *m1 = nir_imax(b, nir_channel(b, t1, 1), nir_imm_int(b, 0));
   nir_push_else(b, NULL);
   nir_pop_if(b, NULL);
   nir_def *p1 = nir_if_phi(b, m1, nir_imm_int(b, 0));
   nir_def *a2 = nir_iadd(b, a1, p1);
   nir_def *keep = nir_ieq_imm(b, s.c, 7); /* independent of any sample */

   nir_def *t2 = txf3d(b, s.c, 32);
   nir_def *n2 = nir_ineg(b, nir_channel(b, t2, 2));
   nir_def *a3 = nir_iadd(b, a2, n2);
   nir_def *a4 = nir_bcsel(b, keep, a3, a2);
   nir_block *b_t0 = def_block(t0), *b_t1 = def_block(t1), *b_t2 = def_block(t2);
   nir_block *b_m1 = def_block(m1), *b_keep = def_block(keep);
   CHECK(b_t0 != b_t2);
   loop_end(&s, a4);

   CHECK(agx_nir_opt_sink_loop_sample_uses(b->shader));
   nir_validate_shader(b->shader, "after");

   CHECK(def_block(t0) == b_t0);
   CHECK(def_block(t1) == b_t1);
   CHECK(def_block(t2) == b_t2);
   CHECK(def_block(n0) == b_t2);
   CHECK(def_block(a1) == b_t2);
   CHECK(def_block(a2) == b_t2);
   CHECK(def_block(a3) == b_t2);
   CHECK(def_block(a4) == b_t2);
   /* A consumer inside the conditional stays: its use is the phi. */
   CHECK(def_block(m1) == b_m1);
   /* Sample-independent ALU keeps its place. */
   CHECK(def_block(keep) == b_keep);
   /* Producers precede their consumers in the target block. */
   CHECK(nir_block_dominates(def_block(n0), def_block(a1)));
   bool seen_n0 = false, ordered = true;
   nir_foreach_instr(instr, b_t2) {
      if (instr == nir_def_instr(n0))
         seen_n0 = true;
      if (instr == nir_def_instr(a1) && !seen_n0)
         ordered = false;
   }
   CHECK(ordered);
   CHECK(!agx_nir_opt_sink_loop_sample_uses(b->shader));
   ralloc_free(b->shader);
}

/* Thirteen 32-bit vec4 samples: the first twelve (96 halves) sink their
 * consumers; the thirteenth's own consumer stays where it was. */
static void
test_budget(void)
{
   struct loop_shader s;
   loop_begin(&s);
   nir_builder *b = &s.b;
   nir_def *neg[13];
   nir_def *acc = s.acc;
   for (unsigned i = 0; i < 13; ++i) {
      nir_def *t = txf3d(b, s.c, 32);
      neg[i] = nir_ineg(b, nir_channel(b, t, 0));
      acc = nir_iadd(b, acc, neg[i]);
      nir_push_if(b, nir_ine_imm(b, s.c, (int)i + 1));
      nir_pop_if(b, NULL);
   }
   nir_block *before[13];
   for (unsigned i = 0; i < 13; ++i)
      before[i] = def_block(neg[i]);
   nir_def *last = txf3d(b, s.c, 32);
   acc = nir_iadd(b, acc, nir_channel(b, last, 3));
   nir_block *tail = def_block(last);
   loop_end(&s, acc);

   CHECK(agx_nir_opt_sink_loop_sample_uses(b->shader));
   nir_validate_shader(b->shader, "after");
   for (unsigned i = 0; i < 12; ++i)
      CHECK(def_block(neg[i]) == tail);
   CHECK(def_block(neg[12]) == before[12]);
   ralloc_free(b->shader);
}

/* No sample in the loop: nothing moves. */
static void
test_no_samples(void)
{
   struct loop_shader s;
   loop_begin(&s);
   nir_builder *b = &s.b;
   nir_def *a1 = nir_iadd(b, s.acc, s.c);
   nir_push_if(b, nir_ine_imm(b, s.c, 0));
   nir_pop_if(b, NULL);
   nir_def *a2 = nir_imul(b, a1, s.c);
   nir_block *b1 = def_block(a1);
   loop_end(&s, a2);
   CHECK(!agx_nir_opt_sink_loop_sample_uses(b->shader));
   CHECK(def_block(a1) == b1);
   ralloc_free(b->shader);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_witness_shape();
   test_budget();
   test_no_samples();
   glsl_type_singleton_decref();
   if (failures) {
      fprintf(stderr, "%u check(s) failed\n", failures);
      return EXIT_FAILURE;
   }
   printf("agx_nir_opt_sink_loop_sample_uses: all checks passed\n");
   return EXIT_SUCCESS;
}
