// N-tuple network value function V(afterstate) for 2048 with 8-fold symmetry sampling
// and optional multi-staging (separate weight sets selected by a board-dependent stage).
#pragma once
#include "board.hpp"
#include <cmath>
#include <cstdlib>
#include <sys/mman.h>
#include <cstdio>
#include <string>
#include <vector>

// Weight tables are allocated 2MB-aligned and, on Linux, backed by transparent huge pages:
// random lookups into ~1GB of tables otherwise miss the TLB almost every time.
inline float* alloc_table(size_t n, bool zero) {
    size_t bytes = (n * sizeof(float) + (2u << 20) - 1) & ~size_t((2u << 20) - 1);
    float* a = static_cast<float*>(aligned_alloc(2u << 20, bytes));
#ifdef __linux__
    madvise(a, bytes, MADV_HUGEPAGE);
#endif
    if (zero) memset(a, 0, bytes);
    return a;
}

struct Pattern {
    std::vector<int> cells;
    int sym[8][8];  // cells under each of the 8 symmetries (only cells.size() used)
};

// Map: cell index under symmetry k. We derive it by transforming a board with a marker.
inline int sym_cell(int c, int k) {
    int r = c / 4, col = c % 4;
    if (k & 4) std::swap(r, col);  // transpose
    if (k & 1) col = 3 - col;      // flip_h
    if (k & 2) r = 3 - r;          // flip_v
    return 4 * r + col;
}

inline uint32_t extract(board_t b, const int* cells, int n) {
    uint32_t idx = 0;
    for (int k = 0; k < n; k++) idx |= uint32_t((b >> (4 * cells[k])) & 0xF) << (4 * k);
    return idx;
}

struct Network {
    std::vector<Pattern> pats;
    int nstages = 1;
    int stage_mode = 0;
    std::vector<float*> w;    // w[stage * npat + p] -> 16^n floats
    std::vector<float*> E, A; // TC learning accumulators (optional)
    bool tc = false;

    static std::vector<std::vector<int>> preset(const std::string& name) {
        if (name.rfind("cells:", 0) == 0) {  // explicit tuples, e.g. "cells:0,1,2,4,5,6;4,5,6,7,8,9"
            std::vector<std::vector<int>> P(1);
            int v = -1;
            for (char ch : name.substr(6) + ";") {
                if (ch >= '0' && ch <= '9') v = (v < 0 ? 0 : v * 10) + (ch - '0');
                else if (ch == ',' || ch == ';') {
                    if (v >= 0) P.back().push_back(v);
                    v = -1;
                    if (ch == ';' && !P.back().empty()) P.emplace_back();
                }
            }
            if (P.back().empty()) P.pop_back();
            return P;
        }
        if (name == "4x6")  // Yeh et al. / Jaskowski
            return {{0, 1, 2, 3, 4, 5}, {4, 5, 6, 7, 8, 9}, {0, 1, 2, 4, 5, 6}, {4, 5, 6, 8, 9, 10}};
        if (name == "8x6")  // NOTE: only 6 distinct tuples under symmetry (2 duplicates); kept for old weights
            return {{0, 1, 2, 3, 4, 5}, {4, 5, 6, 7, 8, 9}, {0, 1, 2, 4, 5, 6}, {4, 5, 6, 8, 9, 10},
                    {0, 1, 4, 5, 8, 9}, {1, 2, 5, 6, 9, 10}, {0, 1, 2, 5, 9, 10}, {0, 1, 5, 9, 13, 14}};
        if (name == "m8x6")  // Matsuzaki's 8 x 6-tuples exactly as in the OTD paper (Fig. 3, TDL2048+ 8x6patt)
            return {{0, 1, 2, 4, 5, 6}, {4, 5, 6, 7, 8, 9}, {0, 1, 2, 3, 4, 5}, {2, 3, 4, 5, 6, 9},
                    {0, 1, 2, 5, 9, 10}, {3, 4, 5, 6, 7, 8}, {1, 3, 4, 5, 6, 7}, {0, 1, 4, 8, 9, 10}};
        if (name == "4x4")
            return {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 4, 5}, {1, 2, 5, 6}};
        fprintf(stderr, "unknown preset %s\n", name.c_str());
        exit(1);
    }

    // Stage of an afterstate. mode 0: single stage.
    // mode 1 (Yeh-style 3 stages, cyclic in 32768): uses the tiles below the largest
    // 32768 "anchor": stage 1 once a 16384 is present, stage 2 once 16384 and 8192 are present.
    // mode 2: 2 stages, stage 1 once a 16384 is present.
    // mode 4: mode 3 plus stage 2 once a 16384 and an 8192 are both present.
    inline int stage(board_t b) const {
        if (nstages == 1) return 0;
        int has14 = 0, has13 = 0;
        for (int i = 0; i < 16; i++) {
            int v = (b >> (4 * i)) & 0xF;
            has14 |= v == 14;
            has13 |= v == 13;
        }
        if (stage_mode == 2) return has14;
        if (stage_mode == 3 || stage_mode == 4) {  // stage 1 once the largest tile is >= 16384
            if (stage_mode == 4 && has14 && has13) return 2;
            for (int i = 0; i < 16; i++) if (((b >> (4 * i)) & 0xF) >= 14) return 1;
            return 0;
        }
        return has14 ? (has13 ? 2 : 1) : 0;
    }

    void init(const std::string& preset_name, int stages, int mode, float init_value, bool use_tc) {
        auto P = preset(preset_name);
        nstages = stages;
        stage_mode = mode;
        tc = use_tc;
        for (auto& cells : P) {
            Pattern p;
            p.cells = cells;
            for (int k = 0; k < 8; k++)
                for (size_t j = 0; j < cells.size(); j++) p.sym[k][j] = sym_cell(cells[j], k);
            pats.push_back(p);
        }
        int nf = int(pats.size()) * 8;
        for (int s = 0; s < nstages; s++)
            for (auto& p : pats) {
                size_t sz = size_t(1) << (4 * p.cells.size());
                float* a = alloc_table(sz, false);
                float v = init_value / nf;
                for (size_t i = 0; i < sz; i++) a[i] = v;
                w.push_back(a);
                if (tc) {
                    E.push_back(alloc_table(sz, true));
                    A.push_back(alloc_table(sz, true));
                }
            }
    }
    size_t size(int p) const { return size_t(1) << (4 * pats[p].cells.size()); }
    int num_features() const { return int(pats.size()) * 8; }

    inline float value(board_t b) const {
        int s = stage(b);
        float v = 0;
        int np = int(pats.size());
        for (int p = 0; p < np; p++) {
            const Pattern& pt = pats[p];
            const float* W = w[s * np + p];
            int n = int(pt.cells.size());
            for (int k = 0; k < 8; k++) v += W[extract(b, pt.sym[k], n)];
        }
        return v;
    }

    // Batched, prefetching evaluation: gather all feature indices of a board first (issuing prefetches),
    // then sum. Evaluating several boards this way overlaps their cache misses (~4x faster).
    struct Feat {
        uint32_t idx[128];
        int n, s;
    };
    inline void features(board_t b, Feat& F, bool prefetch_tc = false) const {
        F.s = stage(b);
        int np = int(pats.size()), f = 0;
        for (int p = 0; p < np; p++) {
            const Pattern& pt = pats[p];
            int n = int(pt.cells.size());
            const float* W = w[F.s * np + p];
            for (int k = 0; k < 8; k++) {
                uint32_t i = extract(b, pt.sym[k], n);
                F.idx[f++] = i;
                __builtin_prefetch(W + i);
                if (prefetch_tc) { __builtin_prefetch(E[F.s * np + p] + i); __builtin_prefetch(A[F.s * np + p] + i); }
            }
        }
        F.n = f;
    }
    inline float sum(const Feat& F) const {
        int np = int(pats.size());
        float v = 0;
        for (int f = 0; f < F.n; f++) v += w[F.s * np + (f >> 3)][F.idx[f]];
        return v;
    }
    inline void update(const Feat& F, float delta, float alpha) {
        int np = int(pats.size());
        for (int f = 0; f < F.n; f++) {
            int t = F.s * np + (f >> 3);
            uint32_t i = F.idx[f];
            if (!tc) w[t][i] += alpha * delta;
            else {
                float rate = A[t][i] == 0 ? 1.0f : std::fabs(E[t][i]) / A[t][i];
                w[t][i] += alpha * rate * delta;
                E[t][i] += delta;
                A[t][i] += std::fabs(delta);
            }
        }
    }
    // values of up to 4 boards
    inline void value_batch(const board_t* bs, int n, float* out) const {
        Feat F[4];
        for (int k = 0; k < n; k++) features(bs[k], F[k]);
        for (int k = 0; k < n; k++) out[k] = sum(F[k]);
    }

    // Apply TD error delta to all active features. alpha is the step size per update (already / nf).
    inline void update(board_t b, float delta, float alpha) {
        int s = stage(b);
        int np = int(pats.size());
        for (int p = 0; p < np; p++) {
            const Pattern& pt = pats[p];
            int n = int(pt.cells.size());
            float* W = w[s * np + p];
            if (!tc) {
                float u = alpha * delta;
                for (int k = 0; k < 8; k++) W[extract(b, pt.sym[k], n)] += u;
            } else {
                float* e = E[s * np + p];
                float* a = A[s * np + p];
                for (int k = 0; k < 8; k++) {
                    uint32_t i = extract(b, pt.sym[k], n);
                    float rate = a[i] == 0 ? 1.0f : std::fabs(e[i]) / a[i];
                    W[i] += alpha * rate * delta;
                    e[i] += delta;
                    a[i] += std::fabs(delta);
                }
            }
        }
    }

    // Weight promotion: copy the weights of the current last stage into the new stages.
    // If init_value is given (>= 0), new stages start optimistic at init_value / #features instead.
    // If opt_unseen > 0 (requires TC state), features never updated so far (A == 0) start at opt_unseen / #features.
    void promote(int stages, int mode, float init_value = -1, float bonus = 0, float opt_unseen = 0) {
        int np = int(pats.size());
        int src = (nstages - 1) * np;
        for (int s = nstages; s < stages; s++)
            for (int p = 0; p < np; p++) {
                float* a = alloc_table(size(p), false);
                if (init_value >= 0) {
                    float v = init_value / num_features();
                    for (size_t i = 0; i < size(p); i++) a[i] = v;
                } else {
                    memcpy(a, w[src + p], size(p) * sizeof(float));
                    float v = bonus / num_features();
                    if (bonus != 0) for (size_t i = 0; i < size(p); i++) a[i] += v;
                    if (opt_unseen > 0 && tc) {
                        float o = opt_unseen / num_features();
                        size_t cnt = 0;
                        for (size_t i = 0; i < size(p); i++) if (A[src + p][i] == 0) { a[i] = o; cnt++; }
                        printf("pattern %d: %zu of %zu features unseen -> optimistic\n", p, cnt, size(p));
                    }
                }
                w.push_back(a);
                if (tc) {  // keep the annealed TC rates of the source stage
                    float* e = alloc_table(size(p), false);
                    float* aa = alloc_table(size(p), false);
                    memcpy(e, E[src + p], size(p) * sizeof(float));
                    memcpy(aa, A[src + p], size(p) * sizeof(float));
                    E.push_back(e);
                    A.push_back(aa);
                }
            }
        nstages = stages;
        stage_mode = mode;
    }

    // Pin all tables in RAM so the OS can neither swap nor compress them (random access to
    // compressed pages costs a decompression fault each time).
    void lock_memory() const {
        size_t ok = 0, fail = 0;
        auto lk = [&](const float* a, size_t n) { (mlock(a, n * sizeof(float)) == 0 ? ok : fail) += n * sizeof(float); };
        int np = int(pats.size());
        for (size_t i = 0; i < w.size(); i++) lk(w[i], size(int(i % np)));
        for (size_t i = 0; i < E.size(); i++) { lk(E[i], size(int(i % np))); lk(A[i], size(int(i % np))); }
        printf("mlock: %.2f GB locked, %.2f GB failed\n", ok / 1e9, fail / 1e9);
        fflush(stdout);
    }

    void disable_tc() {
        for (auto* e : E) free(e);
        for (auto* a : A) free(a);
        E.clear(); A.clear();
        tc = false;
    }

    void enable_tc() {
        if (tc) return;
        tc = true;
        for (size_t i = 0; i < w.size(); i++) {
            size_t sz = size(int(i % pats.size()));
            E.push_back(alloc_table(sz, true));
            A.push_back(alloc_table(sz, true));
        }
    }

    bool save(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) return false;
        int np = int(pats.size());
        fwrite(&np, 4, 1, f);
        for (auto& p : pats) {
            int n = int(p.cells.size());
            fwrite(&n, 4, 1, f);
            fwrite(p.cells.data(), 4, n, f);
        }
        fwrite(&nstages, 4, 1, f);
        fwrite(&stage_mode, 4, 1, f);
        for (size_t i = 0; i < w.size(); i++) fwrite(w[i], 4, size(int(i % np)), f);
        fclose(f);
        return true;
    }
    // TC accumulators are saved separately so learning can resume with its adaptive rates intact.
    bool save_tc(const std::string& path) const {
        if (!tc) return false;
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) return false;
        for (size_t i = 0; i < w.size(); i++) {
            fwrite(E[i], 4, size(int(i % pats.size())), f);
            fwrite(A[i], 4, size(int(i % pats.size())), f);
        }
        fclose(f);
        return true;
    }
    bool load_tc(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        enable_tc();
        for (size_t i = 0; i < w.size(); i++) {
            fread(E[i], 4, size(int(i % pats.size())), f);
            fread(A[i], 4, size(int(i % pats.size())), f);
        }
        fclose(f);
        return true;
    }
    bool load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        int np;
        if (fread(&np, 4, 1, f) != 1) return false;
        pats.clear();
        for (int i = 0; i < np; i++) {
            int n;
            fread(&n, 4, 1, f);
            Pattern p;
            p.cells.resize(n);
            fread(p.cells.data(), 4, n, f);
            for (int k = 0; k < 8; k++)
                for (int j = 0; j < n; j++) p.sym[k][j] = sym_cell(p.cells[j], k);
            pats.push_back(p);
        }
        fread(&nstages, 4, 1, f);
        fread(&stage_mode, 4, 1, f);
        w.clear();
        for (int s = 0; s < nstages; s++)
            for (int p = 0; p < np; p++) {
                float* a = alloc_table(size(p), false);
                fread(a, 4, size(p), f);
                w.push_back(a);
            }
        fclose(f);
        tc = false;
        return true;
    }
};
