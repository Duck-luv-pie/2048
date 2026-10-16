// 4x4 2048 bitboard engine.
// Board: 64-bit, cell i (row-major, i = 4*row + col) stored in nibble i.
// Nibble value v means tile 2^v (0 = empty). Value 15 = 32768.
// Merging two 32768 tiles (65536) cannot be represented; it is reported via a flag.
#pragma once
#include <cstdint>
#include <cstring>
#include <random>

typedef uint64_t board_t;
typedef uint16_t row_t;

enum Dir { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };

namespace tables {
inline row_t row_left[65536], row_right[65536];
inline board_t col_up[65536], col_down[65536];
inline uint32_t score_left[65536], score_right[65536];
inline uint8_t big_left[65536], big_right[65536];  // row produces a 65536 merge
inline bool ready = false;
}

inline board_t unpack_col(row_t r) {
    board_t t = r;
    return (t | (t << 12) | (t << 24) | (t << 36)) & 0x000F000F000F000FULL;
}
inline row_t reverse_row(row_t r) {
    return (r >> 12) | ((r >> 4) & 0x00F0) | ((r << 4) & 0x0F00) | (r << 12);
}
inline board_t transpose(board_t x) {
    board_t a1 = x & 0xF0F00F0FF0F00F0FULL;
    board_t a2 = x & 0x0000F0F00000F0F0ULL;
    board_t a3 = x & 0x0F0F00000F0F0000ULL;
    board_t a = a1 | (a2 << 12) | (a3 >> 12);
    board_t b1 = a & 0xFF00FF0000FF00FFULL;
    board_t b2 = a & 0x00FF00FF00000000ULL;
    board_t b3 = a & 0x00000000FF00FF00ULL;
    return b1 | (b2 >> 24) | (b3 << 24);
}
// mirror columns (reverse each row)
inline board_t flip_h(board_t b) {
    return ((b & 0x000F000F000F000FULL) << 12) | ((b & 0x00F000F000F000F0ULL) << 4) |
           ((b & 0x0F000F000F000F00ULL) >> 4) | ((b & 0xF000F000F000F000ULL) >> 12);
}
// mirror rows (reverse row order)
inline board_t flip_v(board_t b) {
    return ((b & 0x000000000000FFFFULL) << 48) | ((b & 0x00000000FFFF0000ULL) << 16) |
           ((b & 0x0000FFFF00000000ULL) >> 16) | ((b & 0xFFFF000000000000ULL) >> 48);
}

inline void init_tables() {
    using namespace tables;
    if (ready) return;
    for (unsigned r = 0; r < 65536; r++) {
        int line[4] = {int(r & 0xF), int((r >> 4) & 0xF), int((r >> 8) & 0xF), int((r >> 12) & 0xF)};
        // move toward index 0 ("left")
        int out[4] = {0, 0, 0, 0};
        int n = 0;
        uint32_t score = 0;
        uint8_t big = 0;
        int last = 0;  // pending tile that may merge
        for (int i = 0; i < 4; i++) {
            if (!line[i]) continue;
            if (last && last == line[i]) {
                int v = last + 1;
                if (v >= 16) { big = 1; v = 15; score += 65536; } else score += 1u << v;
                out[n++] = v;
                last = 0;
            } else {
                if (last) out[n++] = last;
                last = line[i];
            }
        }
        if (last) out[n++] = last;
        row_t res = row_t(out[0] | (out[1] << 4) | (out[2] << 8) | (out[3] << 12));
        row_t rev = reverse_row(row_t(r));
        row_t revres = reverse_row(res);
        row_left[r] = res;
        score_left[r] = score;
        big_left[r] = big;
        row_right[rev] = revres;
        score_right[rev] = score;
        big_right[rev] = big;
    }
    for (unsigned r = 0; r < 65536; r++) {
        col_up[r] = unpack_col(row_left[r]);
        col_down[r] = unpack_col(row_right[r]);
    }
    ready = true;
}

// Apply move. Returns resulting board (== b if move illegal). Adds reward to *score; sets *big if 65536 formed.
inline board_t do_move(board_t b, int dir, uint32_t* score, int* big) {
    using namespace tables;
    board_t ret = 0;
    uint32_t s = 0;
    int bg = 0;
    if (dir == LEFT || dir == RIGHT) {
        const row_t* T = dir == LEFT ? row_left : row_right;
        const uint32_t* S = dir == LEFT ? score_left : score_right;
        const uint8_t* B = dir == LEFT ? big_left : big_right;
        for (int i = 0; i < 4; i++) {
            row_t r = row_t(b >> (16 * i));
            ret |= board_t(T[r]) << (16 * i);
            s += S[r];
            bg |= B[r];
        }
    } else {
        board_t t = transpose(b);
        const board_t* T = dir == UP ? col_up : col_down;
        const uint32_t* S = dir == UP ? score_left : score_right;
        const uint8_t* B = dir == UP ? big_left : big_right;
        for (int i = 0; i < 4; i++) {
            row_t r = row_t(t >> (16 * i));
            ret |= T[r] << (4 * i);
            s += S[r];
            bg |= B[r];
        }
    }
    if (score) *score = s;
    if (big) *big = bg;
    return ret;
}

inline int count_empty(board_t x) {
    x |= (x >> 2) & 0x3333333333333333ULL;
    x |= (x >> 1);
    x = ~x & 0x1111111111111111ULL;
    return __builtin_popcountll(x);
}
inline uint64_t empty_mask(board_t x) {  // bit 4i set if cell i empty
    x |= (x >> 2) & 0x3333333333333333ULL;
    x |= (x >> 1);
    return ~x & 0x1111111111111111ULL;
}
inline int max_tile(board_t b) {
    int m = 0;
    for (int i = 0; i < 16; i++) { int v = (b >> (4 * i)) & 0xF; if (v > m) m = v; }
    return m;
}
inline int cell(board_t b, int i) { return (b >> (4 * i)) & 0xF; }

struct Rng {
    uint64_t s[4];
    explicit Rng(uint64_t seed = 1) {
        for (int i = 0; i < 4; i++) {
            seed += 0x9E3779B97F4A7C15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            s[i] = z ^ (z >> 31);
        }
    }
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    inline uint32_t below(uint32_t n) { return uint32_t(((next() >> 32) * n) >> 32); }
    inline double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

// Place a random tile (90% 2, 10% 4) on an empty cell. Board must have an empty cell.
inline board_t spawn(board_t b, Rng& rng) {
    uint64_t m = empty_mask(b);
    int n = __builtin_popcountll(m);
    int k = rng.below(n);
    while (k--) m &= m - 1;
    int pos = __builtin_ctzll(m);
    board_t tile = rng.below(10) == 0 ? 2 : 1;
    return b | (tile << pos);
}
inline board_t initial_board(Rng& rng) { return spawn(spawn(0, rng), rng); }

inline bool can_move(board_t b) {
    for (int d = 0; d < 4; d++) if (do_move(b, d, nullptr, nullptr) != b) return true;
    return false;
}
