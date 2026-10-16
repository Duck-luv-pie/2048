// Cross-check the bitboard engine against a naive reference implementation.
#include "board.hpp"
#include <cassert>
#include <cstdio>

static board_t naive_move(board_t b, int dir, uint32_t* score) {
    int g[4][4];
    for (int i = 0; i < 16; i++) g[i / 4][i % 4] = cell(b, i);
    *score = 0;
    for (int line = 0; line < 4; line++) {
        int idx[4][2];
        for (int k = 0; k < 4; k++) {
            int r, c;
            if (dir == LEFT) { r = line; c = k; }
            else if (dir == RIGHT) { r = line; c = 3 - k; }
            else if (dir == UP) { r = k; c = line; }
            else { r = 3 - k; c = line; }
            idx[k][0] = r; idx[k][1] = c;
        }
        int vals[4], n = 0;
        for (int k = 0; k < 4; k++) if (g[idx[k][0]][idx[k][1]]) vals[n++] = g[idx[k][0]][idx[k][1]];
        int out[4] = {0, 0, 0, 0}, m = 0;
        for (int k = 0; k < n; k++) {
            if (k + 1 < n && vals[k] == vals[k + 1]) {
                int v = vals[k] + 1;
                if (v > 15) v = 15;
                *score += 1u << (vals[k] + 1);
                out[m++] = v; k++;
            } else out[m++] = vals[k];
        }
        for (int k = 0; k < 4; k++) g[idx[k][0]][idx[k][1]] = out[k];
    }
    board_t r = 0;
    for (int i = 0; i < 16; i++) r |= board_t(g[i / 4][i % 4]) << (4 * i);
    return r;
}

int main() {
    init_tables();
    Rng rng(42);
    long bad = 0;
    for (int it = 0; it < 2000000; it++) {
        board_t b = 0;
        for (int i = 0; i < 16; i++) {
            int v = rng.below(3) == 0 ? 0 : 1 + rng.below(it % 2 ? 4 : 15);
            b |= board_t(v) << (4 * i);
        }
        for (int d = 0; d < 4; d++) {
            uint32_t s1, s2;
            board_t m1 = do_move(b, d, &s1, nullptr), m2 = naive_move(b, d, &s2);
            if (m1 != m2 || s1 != s2) { if (bad < 5) printf("mismatch dir %d %016llx\n", d, (unsigned long long)b); bad++; }
        }
        // symmetry ops are involutions
        assert(transpose(transpose(b)) == b && flip_h(flip_h(b)) == b && flip_v(flip_v(b)) == b);
    }
    printf("mismatches: %ld\n", bad);
    return bad != 0;
}
