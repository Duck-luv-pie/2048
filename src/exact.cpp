// Exact solution of 2048 on small R x C boards by memoised expectimax over afterstates
// (finite-horizon Bellman equation: the tile sum strictly increases each turn, so the MDP is acyclic).
//
//   V(afterstate) = sum_spawn P(spawn) * max_a [ r(s,a) + V(after(s,a)) ]   (0 if no legal move)
//
// Objectives: "score" = maximise expected final score,
//             "reach K" = maximise probability of ever creating a tile >= 2^K.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef uint64_t board_t;
int R, C, N;
int objective_reach = 0;  // 0 => expected score
std::vector<std::vector<int>> lines[4];  // per direction: list of lines, each a list of cell indices in move order
std::vector<std::vector<int>> syms;      // cell permutations

inline int cell(board_t b, int i) { return (b >> (4 * i)) & 0xF; }

board_t do_move(board_t b, int dir, uint32_t* score) {
    board_t out = b;
    uint32_t s = 0;
    for (auto& ln : lines[dir]) {
        int vals[4], n = 0;
        for (int c : ln) if (cell(b, c)) vals[n++] = cell(b, c);
        int res[4] = {0, 0, 0, 0}, m = 0;
        for (int k = 0; k < n; k++) {
            if (k + 1 < n && vals[k] == vals[k + 1]) { res[m++] = vals[k] + 1; s += 1u << (vals[k] + 1); k++; }
            else res[m++] = vals[k];
        }
        for (size_t k = 0; k < ln.size(); k++) {
            out &= ~(board_t(0xF) << (4 * ln[k]));
            out |= board_t(res[k]) << (4 * ln[k]);
        }
    }
    *score = s;
    return out;
}

board_t canon(board_t b) {
    board_t best = b;
    for (auto& p : syms) {
        board_t t = 0;
        for (int i = 0; i < N; i++) t |= board_t(cell(b, i)) << (4 * p[i]);
        if (t < best) best = t;
    }
    return best;
}

// open addressing hash map board -> double
struct Map {
    std::vector<board_t> keys;
    std::vector<float> vals;  // float is plenty for reporting; keeps memory down
    size_t mask, count = 0;
    explicit Map(int logsz) : keys(size_t(1) << logsz, ~0ULL), vals(size_t(1) << logsz), mask((size_t(1) << logsz) - 1) {}
    static inline uint64_t h(uint64_t x) {
        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
        return x;
    }
    float* find(board_t k) {
        size_t i = h(k) & mask;
        while (keys[i] != ~0ULL) { if (keys[i] == k) return &vals[i]; i = (i + 1) & mask; }
        return nullptr;
    }
    void put(board_t k, float v) {
        size_t i = h(k) & mask;
        while (keys[i] != ~0ULL) { if (keys[i] == k) { vals[i] = v; return; } i = (i + 1) & mask; }
        keys[i] = k; vals[i] = v;
        if (++count * 10 > keys.size() * 8) { fprintf(stderr, "hash table full\n"); exit(1); }
    }
};
Map* memo;
int max_tile_seen = 0;

double V(board_t af);
double max_node(board_t s) {
    double best = 0;
    bool any = false;
    for (int d = 0; d < 4; d++) {
        uint32_t r;
        board_t af = do_move(s, d, &r);
        if (af == s) continue;
        double v;
        if (objective_reach) {
            int mt = 0;
            for (int i = 0; i < N; i++) mt = std::max(mt, cell(af, i));
            v = mt >= objective_reach ? 1.0 : V(af);
        } else v = r + V(af);
        if (!any || v > best) best = v;
        any = true;
    }
    return any ? best : 0.0;
}
double V(board_t af) {
    board_t k = canon(af);
    if (float* p = memo->find(k)) return *p;
    for (int i = 0; i < N; i++) max_tile_seen = std::max(max_tile_seen, cell(af, i));
    double sum = 0;
    int n = 0;
    for (int i = 0; i < N; i++)
        if (!cell(af, i)) {
            n++;
            sum += 0.9 * max_node(af | (board_t(1) << (4 * i))) + 0.1 * max_node(af | (board_t(2) << (4 * i)));
        }
    double v = n ? sum / n : 0.0;  // afterstates always have an empty cell
    memo->put(k, float(v));
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: exact R C [score|reach K] [logHashSize]\n"); return 1; }
    R = atoi(argv[1]); C = atoi(argv[2]); N = R * C;
    int argi = 3;
    if (argc > 3 && std::string(argv[3]) == "reach") { objective_reach = atoi(argv[4]); argi = 5; }
    else if (argc > 3 && std::string(argv[3]) == "score") argi = 4;
    int logsz = argc > argi ? atoi(argv[argi]) : 26;
    memo = new Map(logsz);
    // directions: 0 up,1 down,2 left,3 right; each line listed in the order tiles move toward
    for (int c = 0; c < C; c++) {
        std::vector<int> u, d;
        for (int r = 0; r < R; r++) u.push_back(r * C + c), d.push_back((R - 1 - r) * C + c);
        lines[0].push_back(u); lines[1].push_back(d);
    }
    for (int r = 0; r < R; r++) {
        std::vector<int> l, rt;
        for (int c = 0; c < C; c++) l.push_back(r * C + c), rt.push_back(r * C + C - 1 - c);
        lines[2].push_back(l); lines[3].push_back(rt);
    }
    // symmetries: identity, flips; plus transposes for square boards
    for (int k = 0; k < (R == C ? 8 : 4); k++) {
        std::vector<int> p(N);
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++) {
                int rr = r, cc = c;
                if (k & 1) cc = C - 1 - cc;
                if (k & 2) rr = R - 1 - rr;
                int idx = rr * C + cc;
                if (k & 4) idx = cc * C + rr;  // transpose (square only)
                p[r * C + c] = idx;
            }
        syms.push_back(p);
    }
    // initial position: two random tiles on an empty board
    double total = 0, cnt = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            for (int a = 1; a <= 2; a++)
                for (int b = 1; b <= 2; b++) {
                    double p = (a == 1 ? 0.9 : 0.1) * (b == 1 ? 0.9 : 0.1);
                    board_t s = (board_t(a) << (4 * i)) | (board_t(b) << (4 * j));
                    total += p * max_node(s);
                    cnt += p;
                }
        }
    double v = total / cnt;
    if (objective_reach)
        printf("%dx%d  optimal P(reach %d) = %.6f   [%zu canonical afterstates, max tile seen %d]\n", R, C,
               1 << objective_reach, v, memo->count, 1 << max_tile_seen);
    else
        printf("%dx%d  optimal expected score = %.4f   [%zu canonical afterstates, max tile %d]\n", R, C, v,
               memo->count, 1 << max_tile_seen);
}
