// GPU TD(0) trainer for 2048 n-tuple networks.
//
// Training is bound by random-access memory bandwidth (every move gathers ~320 scattered weights
// from a ~1GB table), which GPUs serve far better than CPUs. Each CUDA thread plays its own game
// with online (forward) TD(0) afterstate updates applied with atomicAdd:
//   after choosing afterstate s'_{t+1} (reward r_{t+1}):  V(s'_t) += alpha * (r_{t+1} + V(s'_{t+1}) - V(s'_t))
// Episodes start from --starts states (or a restart buffer of earlier stage>=1 afterstates).
// Weights use the same file format as ntuple.hpp so CPU tools can load the result.
#include "ntuple.hpp"
#include <chrono>
#include <cuda_runtime.h>
#include <thread>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int MAXP = 16, MAXC = 7;  // up to 16 tuples per stage, each of up to 7 cells
__constant__ int8_t c_sym[MAXP][8][MAXC];
__constant__ int c_ncell[MAXP];
__constant__ unsigned long long c_off[MAXP];  // offset of tuple p's table within a stage
__constant__ unsigned long long c_stage_sz;   // weights per stage
__constant__ int c_np;
__constant__ int c_stage_mode;
__constant__ float c_alpha;
__constant__ float c_restart_prob, c_push_prob, c_fresh_prob;
__constant__ int c_train_lo, c_train_hi;  // only stages in [lo, hi] are updated
// Goal mode (c_goal > 0): V(s) = P(creating a tile >= 2^goal). Reward 1 for the move that creates it
// (episode ends), 0 otherwise. c_stop_after: end episodes whose afterstate is in a stage above it,
// bootstrapping from that stage's (frozen) value, so early-stage training never plays the late game.
__constant__ int c_goal, c_stop_after;
__constant__ float c_goal_bonus;  // > 0: keep score rewards and add this bonus for creating the goal tile

struct GState {
    uint64_t board, prev, rng;
    float vprev;
    float score;
    int moves;
    int kind;  // 0 = from start state, 1 = from restart buffer, 2 = fresh game from the empty board
    int maxt;
};
struct GStats {
    unsigned long long games, moves, reach[18];
    double score;
};

__device__ __forceinline__ uint64_t xs(uint64_t& s) {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return s * 2685821657736338717ULL;
}
__device__ __forceinline__ uint32_t below(uint64_t& s, uint32_t n) { return uint32_t(((xs(s) >> 32) * n) >> 32); }

__device__ __forceinline__ uint64_t d_transpose(uint64_t x) {
    uint64_t a1 = x & 0xF0F00F0FF0F00F0FULL, a2 = x & 0x0000F0F00000F0F0ULL, a3 = x & 0x0F0F00000F0F0000ULL;
    uint64_t a = a1 | (a2 << 12) | (a3 >> 12);
    uint64_t b1 = a & 0xFF00FF0000FF00FFULL, b2 = a & 0x00FF00FF00000000ULL, b3 = a & 0x00000000FF00FF00ULL;
    return b1 | (b2 >> 24) | (b3 << 24);
}
__device__ __forceinline__ uint64_t d_unpack_col(uint32_t r) {
    uint64_t t = r;
    return (t | (t << 12) | (t << 24) | (t << 36)) & 0x000F000F000F000FULL;
}
struct DTables {
    const uint16_t *left, *right;
    const uint32_t *sleft, *sright;
    const uint8_t *bleft, *bright;
};
__device__ uint64_t d_move(uint64_t b, int dir, const DTables& T, uint32_t* score, int* big) {
    uint64_t ret = 0;
    uint32_t s = 0;
    int bg = 0;
    if (dir >= 2) {
        const uint16_t* R = dir == 2 ? T.left : T.right;
        const uint32_t* S = dir == 2 ? T.sleft : T.sright;
        const uint8_t* B = dir == 2 ? T.bleft : T.bright;
        for (int i = 0; i < 4; i++) {
            uint32_t r = uint32_t(b >> (16 * i)) & 0xFFFF;
            ret |= uint64_t(__ldg(R + r)) << (16 * i);
            s += __ldg(S + r);
            bg |= __ldg(B + r);
        }
    } else {
        uint64_t t = d_transpose(b);
        const uint16_t* R = dir == 0 ? T.left : T.right;
        const uint32_t* S = dir == 0 ? T.sleft : T.sright;
        const uint8_t* B = dir == 0 ? T.bleft : T.bright;
        for (int i = 0; i < 4; i++) {
            uint32_t r = uint32_t(t >> (16 * i)) & 0xFFFF;
            ret |= d_unpack_col(__ldg(R + r)) << (4 * i);
            s += __ldg(S + r);
            bg |= __ldg(B + r);
        }
    }
    *score = s;
    *big = bg;
    return ret;
}
__device__ __forceinline__ uint64_t d_empty_mask(uint64_t x) {
    x |= (x >> 2) & 0x3333333333333333ULL;
    x |= (x >> 1);
    return ~x & 0x1111111111111111ULL;
}
__device__ __forceinline__ uint64_t d_spawn(uint64_t b, uint64_t& rng) {
    uint64_t m = d_empty_mask(b);
    int n = __popcll(m);
    int k = below(rng, n);
    while (k--) m &= m - 1;
    int pos = __ffsll((long long)m) - 1;
    uint64_t tile = below(rng, 10) == 0 ? 2 : 1;
    return b | (tile << pos);
}
__device__ __forceinline__ int d_maxtile(uint64_t b) {
    int m = 0;
    for (int i = 0; i < 16; i++) m = max(m, int((b >> (4 * i)) & 0xF));
    return m;
}
__device__ __forceinline__ int d_stage(uint64_t b) {
    if (c_stage_mode != 3 && c_stage_mode != 4) return 0;
    int s = 0, has13 = 0, has14 = 0;
    for (int i = 0; i < 16; i++) {
        int v = int((b >> (4 * i)) & 0xF);
        s |= v >= 14;
        has13 |= v == 13;
        has14 |= v == 14;
    }
    return c_stage_mode == 4 && has13 && has14 ? 2 : s;
}
__device__ __forceinline__ uint32_t d_extract(uint64_t b, const int8_t* cells, int n) {
    uint32_t idx = 0;
#pragma unroll
    for (int k = 0; k < MAXC; k++)
        if (k < n) idx |= uint32_t((b >> (4 * cells[k])) & 0xF) << (4 * k);
    return idx;
}
__device__ float d_value(const float* __restrict__ W, uint64_t b) {
    const float* base = W + size_t(d_stage(b)) * c_stage_sz;
    float v = 0;
    for (int p = 0; p < c_np; p++) {
        const float* Wp = base + c_off[p];
        const int n = c_ncell[p];
#pragma unroll
        for (int k = 0; k < 8; k++) v += __ldcg(Wp + d_extract(b, c_sym[p][k], n));
    }
    return v;
}
// Synchronous mini-batch TD: during a step every game adds its TD error into D (and a hit into C)
// for each active feature; k_apply then updates each touched weight exactly once:
//   W += alpha * D * min(1, cap / C)
// so a weight hit by thousands of games in the same step cannot overshoot.
__constant__ float c_cap;
// D[i] packs (hit count << 43) + fixed-point TD-error sum (scale 2^4, signed, |sum| < 2^42) so a
// single 64-bit atomic records both; decode: c = round(T / 2^43), sum = T - c * 2^43.
// One step adds at most 8 hits per game to a weight, so the count fits for fewer than 2^17 games
// (a 16-bit count overflowed when many games shared a feature, e.g. near-empty early boards).
constexpr double DSCALE = 16.0;
constexpr int CSHIFT = 43;
__device__ void d_accumulate(unsigned long long* D, uint64_t b, float delta) {
    int stg = d_stage(b);
    if (stg < c_train_lo || stg > c_train_hi) return;
    size_t base = size_t(stg) * c_stage_sz;
    long long q = __double2ll_rn(double(delta) * DSCALE);
    unsigned long long add = (1ULL << CSHIFT) + (unsigned long long)q;
    for (int p = 0; p < c_np; p++) {
        size_t off = base + c_off[p];
        const int n = c_ncell[p];
#pragma unroll
        for (int k = 0; k < 8; k++) atomicAdd(D + off + d_extract(b, c_sym[p][k], n), add);
    }
}
__global__ void k_apply(const uint64_t* U, int n, float* W, unsigned long long* D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    uint64_t b = U[t];
    if (!b) return;
    int stg = d_stage(b);
    if (stg < c_train_lo || stg > c_train_hi) return;
    size_t base = size_t(stg) * c_stage_sz;
    for (int p = 0; p < c_np; p++) {
        size_t off = base + c_off[p];
        const int n = c_ncell[p];
        for (int k = 0; k < 8; k++) {
            size_t i = off + d_extract(b, c_sym[p][k], n);
            unsigned long long T = atomicExch(D + i, 0ULL);
            if (T) {  // exactly one thread claims each touched weight
                long long tv = (long long)T;
                long long c = (tv + (1LL << (CSHIFT - 1))) >> CSHIFT;
                double sum = double(tv - (c << CSHIFT)) / DSCALE;
                W[i] += c_alpha * float(sum) * fminf(1.f, c_cap / float(c));
            }
        }
    }
}

__device__ void new_episode(GState& g, const uint64_t* starts, int nstarts, const uint64_t* rbuf,
                            const unsigned long long* rcount, unsigned long long rcap) {
    unsigned long long nb = *rcount;
    if (c_fresh_prob > 0 && (xs(g.rng) >> 40) * (1.0f / 16777216.0f) < c_fresh_prob) {
        g.board = d_spawn(d_spawn(0, g.rng), g.rng);
        g.kind = 2;
    } else if (nb >= 10000 && (xs(g.rng) >> 40) * (1.0f / 16777216.0f) < c_restart_prob) {
        unsigned long long n = nb < rcap ? nb : rcap;
        g.board = d_spawn(rbuf[(xs(g.rng) >> 11) % n], g.rng);
        g.kind = 1;
    } else {
        g.board = starts[below(g.rng, nstarts)];
        g.kind = 0;
    }
    g.prev = 0;
    g.vprev = 0;
    g.score = 0;
    g.moves = 0;
    g.maxt = d_maxtile(g.board);
}

__global__ void k_debug(GState* S, float* W, DTables T) {
    GState g = S[0];
    printf("board %016llx\n", (unsigned long long)g.board);
    for (int d = 0; d < 4; d++) {
        uint32_t r; int big;
        uint64_t af = d_move(g.board, d, T, &r, &big);
        printf("dir %d af %016llx r %u v %f stage %d\n", d, (unsigned long long)af, r, af != g.board ? d_value(W, af) : 0.f, d_stage(af));
    }
}

__global__ void k_nancount(const float* W, size_t n, unsigned long long* out, float* amax) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    unsigned long long c = 0;
    float m = 0;
    for (; i < n; i += stride) { float v = W[i]; if (!isfinite(v)) c++; else m = fmaxf(m, fabsf(v)); }
    atomicAdd(out, c);
    atomicMax((int*)amax, __float_as_int(m));
}

__global__ void k_init(GState* S, int n, const uint64_t* starts, int nstarts, uint64_t seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    GState g;
    g.rng = seed ^ (0x9E3779B97F4A7C15ULL * (i + 1));
    for (int k = 0; k < 4; k++) xs(g.rng);
    g.board = starts[below(g.rng, nstarts)];
    g.prev = 0; g.vprev = 0; g.score = 0; g.moves = 0; g.kind = 0; g.maxt = d_maxtile(g.board);
    S[i] = g;
}

__global__ void k_train(GState* S, int n, int steps, const float* W, unsigned long long* D, uint64_t* U, DTables T,
                        const uint64_t* starts, int nstarts, uint64_t* rbuf, unsigned long long* rcount,
                        unsigned long long rcap, GStats* st) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    GState g = S[i];
    U[i] = 0;
    for (int step = 0; step < steps; step++) {
        uint64_t b = g.board;
        uint64_t best_af = 0;
        float best = -1e30f, best_v = 0;
        uint32_t best_r = 0;
        int best_big = 0;
        bool best_goal = false;
        for (int d = 0; d < 4; d++) {
            uint32_t r; int big;
            uint64_t af = d_move(b, d, T, &r, &big);
            if (af == b) continue;
            bool goal = c_goal > 0 && (big || d_maxtile(af) >= c_goal);
            float v = goal ? 0.f : d_value(const_cast<float*>(W), af);
            float rr = c_goal > 0 ? (c_goal_bonus > 0 ? float(r) + (goal ? c_goal_bonus : 0.f) : (goal ? 1.f : 0.f)) : float(r);
            if (rr + v > best) { best = rr + v; best_af = af; best_r = r; best_v = v; best_big = big; best_goal = goal; }
        }
        bool terminal = best == -1e30f;
        if (g.prev) {  // TD error of the previous afterstate (weights are frozen during the step)
            float target = terminal ? 0.f : best;  // best = reward + value of the chosen afterstate (value 0 if goal)
            d_accumulate(D, g.prev, target - g.vprev);
            U[i] = g.prev;
        }
        bool stop = !terminal && c_stop_after >= 0 && d_stage(best_af) > c_stop_after;
        if (terminal || best_big || best_goal || stop) {
            if (best_big) g.maxt = 16;
            if (!terminal) g.maxt = max(g.maxt, d_maxtile(best_af));
            if (g.kind != 1) {
                GStats* sp = st + (g.kind == 2 ? 1 : 0);
                atomicAdd(&sp->games, 1ULL);
                atomicAdd(&sp->moves, (unsigned long long)g.moves);
                atomicAdd(&sp->score, double(g.score + (best_big ? best_r : 0)));
                for (int k = 1; k <= g.maxt; k++) atomicAdd(&sp->reach[k], 1ULL);
            }
            new_episode(g, starts, nstarts, rbuf, rcount, rcap);
            continue;
        }
        g.score += best_r;
        g.moves++;
        g.maxt = max(g.maxt, d_maxtile(best_af));
        g.prev = best_af;
        g.vprev = best_v;
        if (c_push_prob > 0 && d_stage(best_af) >= 1 && (xs(g.rng) >> 40) * (1.0f / 16777216.0f) < c_push_prob) {
            unsigned long long slot = atomicAdd(rcount, 1ULL);
            rbuf[slot % rcap] = best_af;
        }
        g.board = d_spawn(best_af, g.rng);
    }
    S[i] = g;
}

int main(int argc, char** argv) {
    init_tables();
    std::string load, out = "weights/gpu.bin", startsf;
    int nthreads = 1 << 16, steps = 256, block = 128;
    float alpha = 0.1f, restart_prob = 0.6f, push_prob = 1.0f / 256, cap = 4, fresh_prob = 0;
    int train_lo = 0, train_hi = 99, goal = 0, stop_after = -1;
    float goal_bonus = 0;
    double report_secs = 60, save_secs = 900, run_secs = 1e12;
    unsigned long long rcap = 1 << 22;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (k == "--load") load = nx();
        else if (k == "--out") out = nx();
        else if (k == "--starts") startsf = nx();
        else if (k == "--games-parallel") nthreads = std::stoi(nx());
        else if (k == "--steps") steps = std::stoi(nx());
        else if (k == "--block") block = std::stoi(nx());
        else if (k == "--alpha") alpha = std::stof(nx());
        else if (k == "--restart-prob") restart_prob = std::stof(nx());
        else if (k == "--push-prob") push_prob = std::stof(nx());
        else if (k == "--cap") cap = std::stof(nx());
        else if (k == "--fresh-prob") fresh_prob = std::stof(nx());
        else if (k == "--train-stages") { train_lo = std::stoi(nx()); train_hi = std::stoi(nx()); }
        else if (k == "--goal") goal = std::stoi(nx());
        else if (k == "--goal-bonus") goal_bonus = std::stof(nx());
        else if (k == "--stop-after-stage") stop_after = std::stoi(nx());
        else if (k == "--report-secs") report_secs = std::stod(nx());
        else if (k == "--save-secs") save_secs = std::stod(nx());
        else if (k == "--run-secs") run_secs = std::stod(nx());
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 1; }
    }
    if (nthreads >= (1 << 17)) { fprintf(stderr, "--games-parallel must be < 131072 (hit-count field)\n"); return 1; }
    Network net;
    if (!net.load(load)) { fprintf(stderr, "load failed\n"); return 1; }
    int np = int(net.pats.size());
    for (auto& p : net.pats) if (p.cells.size() > MAXC) { fprintf(stderr, "tuples of at most %d cells supported\n", MAXC); return 1; }
    if (np > MAXP) { fprintf(stderr, "too many patterns\n"); return 1; }
    printf("loaded %s: %d patterns, %d stages (mode %d)\n", load.c_str(), np, net.nstages, net.stage_mode);
    std::vector<uint64_t> starts;
    {
        FILE* f = fopen(startsf.c_str(), "rb");
        uint64_t b;
        while (f && fread(&b, 8, 1, f) == 1) starts.push_back(b);
        if (f) fclose(f);
    }
    if (starts.empty()) { fprintf(stderr, "need --starts\n"); return 1; }
    printf("%zu start states\n", starts.size());

    int8_t sym[MAXP][8][MAXC] = {};
    int ncell[MAXP] = {};
    unsigned long long off[MAXP] = {}, stage_sz = 0;
    for (int p = 0; p < np; p++) {
        ncell[p] = int(net.pats[p].cells.size());
        off[p] = stage_sz;
        stage_sz += net.size(p);
        for (int k = 0; k < 8; k++)
            for (int j = 0; j < ncell[p]; j++) sym[p][k][j] = int8_t(net.pats[p].sym[k][j]);
    }
    CK(cudaMemcpyToSymbol(c_sym, sym, sizeof sym));
    CK(cudaMemcpyToSymbol(c_ncell, ncell, sizeof ncell));
    CK(cudaMemcpyToSymbol(c_off, off, sizeof off));
    CK(cudaMemcpyToSymbol(c_stage_sz, &stage_sz, sizeof stage_sz));
    CK(cudaMemcpyToSymbol(c_np, &np, sizeof np));
    CK(cudaMemcpyToSymbol(c_stage_mode, &net.stage_mode, sizeof(int)));
    float a = alpha / (np * 8);
    CK(cudaMemcpyToSymbol(c_alpha, &a, sizeof a));
    CK(cudaMemcpyToSymbol(c_restart_prob, &restart_prob, sizeof(float)));
    CK(cudaMemcpyToSymbol(c_push_prob, &push_prob, sizeof(float)));
    CK(cudaMemcpyToSymbol(c_cap, &cap, sizeof(float)));
    CK(cudaMemcpyToSymbol(c_fresh_prob, &fresh_prob, sizeof(float)));
    CK(cudaMemcpyToSymbol(c_train_lo, &train_lo, sizeof(int)));
    CK(cudaMemcpyToSymbol(c_train_hi, &train_hi, sizeof(int)));
    CK(cudaMemcpyToSymbol(c_goal, &goal, sizeof(int)));
    CK(cudaMemcpyToSymbol(c_stop_after, &stop_after, sizeof(int)));
    CK(cudaMemcpyToSymbol(c_goal_bonus, &goal_bonus, sizeof(float)));
    if (goal && goal_bonus > 0) printf("goal bonus %.0f on top of score rewards; episode ends at the goal\n", goal_bonus);
    if (goal) printf("goal mode: V = P(tile >= %d)%s\n", 1 << goal, stop_after >= 0 ? ", episodes stop at the next stage" : "");
    printf("training stages %d..%d, fresh-prob %.2f, restart-prob %.2f\n", train_lo, train_hi, fresh_prob, restart_prob);

    size_t ntab = net.w.size(), nw = size_t(net.nstages) * stage_sz;
    auto toff = [&](size_t t) { return size_t(t / np) * stage_sz + off[t % np]; };  // table t = stage * np + p
    printf("%zu weights (%.2f GB)\n", nw, nw * 4.0 / 1e9);
    float* dW;
    CK(cudaMalloc(&dW, nw * sizeof(float)));
    for (size_t t = 0; t < ntab; t++) CK(cudaMemcpy(dW + toff(t), net.w[t], net.size(int(t % np)) * sizeof(float), cudaMemcpyHostToDevice));

    unsigned long long* dD;
    CK(cudaMalloc(&dD, nw * sizeof(unsigned long long)));
    CK(cudaMemset(dD, 0, nw * sizeof(unsigned long long)));
    DTables T;
    {
        uint16_t *l, *r; uint32_t *sl, *sr; uint8_t *bl, *br;
        CK(cudaMalloc(&l, 65536 * 2)); CK(cudaMalloc(&r, 65536 * 2));
        CK(cudaMalloc(&sl, 65536 * 4)); CK(cudaMalloc(&sr, 65536 * 4));
        CK(cudaMalloc(&bl, 65536)); CK(cudaMalloc(&br, 65536));
        CK(cudaMemcpy(l, tables::row_left, 65536 * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(r, tables::row_right, 65536 * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(sl, tables::score_left, 65536 * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(sr, tables::score_right, 65536 * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(bl, tables::big_left, 65536, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(br, tables::big_right, 65536, cudaMemcpyHostToDevice));
        T = {l, r, sl, sr, bl, br};
    }
    uint64_t *dStarts, *dR;
    unsigned long long* dRc;
    GState* dS;
    GStats* dSt;
    CK(cudaMalloc(&dStarts, starts.size() * 8));
    CK(cudaMemcpy(dStarts, starts.data(), starts.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dR, rcap * 8));
    CK(cudaMalloc(&dRc, 8));
    CK(cudaMemset(dRc, 0, 8));
    CK(cudaMalloc(&dS, size_t(nthreads) * sizeof(GState)));
    uint64_t* dU;
    CK(cudaMalloc(&dU, size_t(nthreads) * 8));
    CK(cudaMalloc(&dSt, 2 * sizeof(GStats)));
    CK(cudaMemset(dSt, 0, 2 * sizeof(GStats)));
    int grid = (nthreads + block - 1) / block;
    k_init<<<grid, block>>>(dS, nthreads, dStarts, int(starts.size()), uint64_t(time(nullptr)));
    CK(cudaDeviceSynchronize());
    if (getenv("G2048_DEBUG")) { k_debug<<<1, 1>>>(dS, dW, T); CK(cudaDeviceSynchronize()); }
    printf("%d parallel games, %d steps per launch\n", nthreads, steps);
    fflush(stdout);

    auto t0 = std::chrono::steady_clock::now(), last_rep = t0, last_save = t0;
    unsigned long long total_games = 0, total_moves_launched = 0;
    while (true) {
        static int dbg_steps = getenv("G2048_NANDBG") ? atoi(getenv("G2048_NANDBG")) : 0;
        static unsigned long long* dN = nullptr;
        static float* dM = nullptr;
        if (dbg_steps && !dN) { CK(cudaMalloc(&dN, 8)); CK(cudaMalloc(&dM, 4)); }
        for (int st = 0; st < steps; st++) {
            k_train<<<grid, block>>>(dS, nthreads, 1, dW, dD, dU, T, dStarts, int(starts.size()), dR, dRc, rcap, dSt);
            k_apply<<<grid, block>>>(dU, nthreads, dW, dD);
            if (dbg_steps > 0) {
                static int sidx = 0;
                sidx++;
                CK(cudaMemset(dN, 0, 8)); CK(cudaMemset(dM, 0, 4));
                k_nancount<<<1024, 256>>>(dW, nw, dN, dM);
                unsigned long long nn; float mm;
                CK(cudaMemcpy(&nn, dN, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&mm, dM, 4, cudaMemcpyDeviceToHost));
                if (sidx <= 5 || sidx % 50 == 0 || nn) printf("step %d: nonfinite %llu, max|w| %g\n", sidx, nn, mm);
                if (nn || sidx >= dbg_steps) exit(0);
            }
        }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        total_moves_launched += (unsigned long long)nthreads * steps;
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now - t0).count();
        if (std::chrono::duration<double>(now - last_rep).count() >= report_secs) {
            GStats sv[2];
            CK(cudaMemcpy(sv, dSt, sizeof sv, cudaMemcpyDeviceToHost));
            CK(cudaMemset(dSt, 0, 2 * sizeof(GStats)));
            std::time_t tt = std::time(nullptr);
            char buf[32];
            strftime(buf, sizeof buf, "%H:%M:%S", localtime(&tt));
            for (int q = 0; q < 2; q++) {
                GStats& s = sv[q];
                if (!s.games) continue;
                if (q == 0) total_games += s.games;
                printf("[%s] %.0fs %.1fM moves/s %s games %llu avg %.0f moves/game %.0f | 8192 %.1f%% 16384 %.1f%% 32768 %.2f%% 65536 %.3f%%\n",
                       buf, el, total_moves_launched / el / 1e6, q == 0 ? "from-16384" : "FULL-GAME", s.games,
                       s.score / s.games, double(s.moves) / s.games, 100.0 * s.reach[13] / s.games,
                       100.0 * s.reach[14] / s.games, 100.0 * s.reach[15] / s.games, 100.0 * s.reach[16] / s.games);
            }
            fflush(stdout);
            last_rep = now;
        }
        bool done = el >= run_secs;
        if (done || std::chrono::duration<double>(now - last_save).count() >= save_secs) {
            for (size_t t = 0; t < ntab; t++) CK(cudaMemcpy(net.w[t], dW + toff(t), net.size(int(t % np)) * sizeof(float), cudaMemcpyDeviceToHost));
            net.save(out);
            printf("  [save] %s\n", out.c_str());
            fflush(stdout);
            last_save = now;
        }
        if (done) break;
    }
}
