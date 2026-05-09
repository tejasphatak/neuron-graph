/* Substrate-LLM inner-loop C kernel — Phase C of original architecture spec.
 *
 * Replaces train_ngram_epoch_jit_negsample with native C. Same algorithm:
 *   For each (ctx, target) pair:
 *     - Compute scores[V] = sum_k decay_powers[k] * W[ctx_rows[k], :]
 *     - Predict = argmax(scores)
 *     - If wrong: delta[ctx_rows, target] += w; delta[ctx_rows, pred] -= w
 *     - Negative sampling: weaken delta[ctx_rows, neg_target] -= w * neg_scale
 *
 * Build: gcc -O3 -march=native -fPIC -shared -fopenmp \
 *           -o _inner_loop.so _inner_loop.c
 *
 * Removes the venv+numba dependency. Single shared object, ctypes-bound.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* Returns: 0 on success. Updates n_pairs_out and n_correct_out. */
int inner_loop_negsample(
    float *W,                       /* [V, V] row-major */
    float *delta,                   /* [V, V] row-major, accumulator */
    int64_t V,
    int64_t *seq_rows,              /* flat array of all sequences' row ids (-1 = invalid) */
    int64_t *seq_offsets,           /* [n_seq+1] offsets into seq_rows */
    int64_t n_seq,
    float *decay_powers,            /* [ctx_window] */
    int ctx_window,
    int neg_samples,
    int64_t *neg_targets,           /* pre-generated random negative target rows */
    float neg_scale,
    int64_t *n_pairs_out,
    int64_t *n_correct_out
) {
    /* Per-pair scratch — score vector */
    float *scores = (float*)malloc(V * sizeof(float));
    if (!scores) return -1;

    int64_t n_pairs = 0;
    int64_t n_correct = 0;
    int64_t neg_idx = 0;

    for (int64_t s = 0; s < n_seq; s++) {
        int64_t start = seq_offsets[s];
        int64_t end = seq_offsets[s + 1];
        int64_t L = end - start;
        if (L < 2) continue;

        for (int64_t i = 0; i < L - 1; i++) {
            int64_t target = seq_rows[start + i + 1];
            if (target < 0) continue;

            /* Build scores */
            memset(scores, 0, V * sizeof(float));
            int ctx_count = 0;
            for (int k = 0; k < ctx_window; k++) {
                int64_t j = i - k;
                if (j < 0) break;
                int64_t ctx = seq_rows[start + j];
                if (ctx < 0) continue;
                float w = decay_powers[k];
                float *Wrow = W + ctx * V;
                for (int64_t v = 0; v < V; v++) {
                    scores[v] += w * Wrow[v];
                }
                ctx_count++;
            }
            if (ctx_count == 0) continue;

            /* argmax */
            float best_score = scores[0];
            int64_t pred = 0;
            for (int64_t v = 1; v < V; v++) {
                if (scores[v] > best_score) {
                    best_score = scores[v];
                    pred = v;
                }
            }
            int no_signal = (best_score <= 0.0f);

            if (pred == target && !no_signal) {
                n_correct++;
            } else {
                /* Standard perceptron update */
                for (int k = 0; k < ctx_window; k++) {
                    int64_t j = i - k;
                    if (j < 0) break;
                    int64_t ctx = seq_rows[start + j];
                    if (ctx < 0) continue;
                    float w = decay_powers[k];
                    delta[ctx * V + target] += w;
                    if (!no_signal) {
                        delta[ctx * V + pred] -= w;
                    }
                }
            }

            /* Negative sampling */
            for (int ns = 0; ns < neg_samples; ns++) {
                int64_t neg_t = neg_targets[neg_idx++];
                if (neg_t == target) continue;
                for (int k = 0; k < ctx_window; k++) {
                    int64_t j = i - k;
                    if (j < 0) break;
                    int64_t ctx = seq_rows[start + j];
                    if (ctx < 0) continue;
                    float w = decay_powers[k];
                    delta[ctx * V + neg_t] -= w * neg_scale;
                }
            }

            n_pairs++;
        }
    }

    *n_pairs_out = n_pairs;
    *n_correct_out = n_correct;
    free(scores);
    return 0;
}
