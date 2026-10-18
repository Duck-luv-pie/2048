// Evaluate a trained n-tuple network as the leaf evaluator of an expectimax search.
// Games run in parallel; prints per-milestone success rates with 95% confidence intervals.
#include "ntuple.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

struct Opts {
    std::string weights;
    int games = 1000, threads = 14, depth = 1, seed = 2048;
    int adaptive = 0;      // if >0: extra depth when the board has <= adaptive empty cells
    std::vector<std::pair<int, int>> deep;  // (max empties, min depth): deeper search on crowded boards
    float cprob = 0.0f;    // probability cutoff for chance nodes
    std::string dump;      // write boards entering each stage (for staged training)
    std::string log;       // per-game CSV
    int dump_mode = 1;     // stage definition used for --dump (1: Yeh 3-stage, 3: first 16384)
    std::string starts;    // if set, game g starts from start state g (cycled) instead of an empty board
    bool trace = false;    // stream every move of every game to stdout (for the live viewer)
    int goal = 0;          // goal mode: the network predicts P(tile >= 2^goal); the game ends when it is made
    float goal_bonus = 0;  // with --goal: keep score rewards and add this bonus for making the goal tile
    bool verbose = false;
    bool heur = false;
    bool downgrade = false;     // use the hand-crafted corner/monotonicity heuristic instead of the network
} O;

Network net;
std::vector<board_t> start_states;

// Hand-crafted heuristic (nneonneo-style): rewards empty cells, adjacent equal tiles and
// monotone rows/columns, penalises large unmerged tiles. Applied to every row and column.
float heur_row[65536];
void init_heur() {
    for (unsigned r = 0; r < 65536; r++) {
        int line[4] = {int(r & 0xF), int((r >> 4) & 0xF), int((r >> 8) & 0xF), int((r >> 12) & 0xF)};
        double sum = 0;
        int empty = 0, merges = 0, prev = 0, counter = 0;
        for (int i = 0; i < 4; i++) {
            int rank = line[i];
            sum += std::pow(rank, 3.5);
            if (rank == 0) empty++;
            else {
                if (prev == rank) counter++;
                else if (counter > 0) { merges += 1 + counter; counter = 0; }
                prev = rank;
            }
        }
        if (counter > 0) merges += 1 + counter;
        double ml = 0, mr = 0;
        for (int i = 1; i < 4; i++) {
            if (line[i - 1] > line[i]) ml += std::pow(line[i - 1], 4) - std::pow(line[i], 4);
            else mr += std::pow(line[i], 4) - std::pow(line[i - 1], 4);
        }
        heur_row[r] = float(200000 + 270.0 * empty + 700.0 * merges - 47.0 * std::min(ml, mr) - 11.0 * sum);
    }
}
inline float heur_value(board_t b) {
    board_t t = transpose(b);
    float v = 0;
    for (int i = 0; i < 4; i++) v += heur_row[(b >> (16 * i)) & 0xFFFF] + heur_row[(t >> (16 * i)) & 0xFFFF];
    return v;
}
// Tile downgrading (Guei et al.), applied to the search root only: if the board holds a 32768 tile
// and some tile 2^k below the largest tile is missing (take the largest such k), every tile above 2^k
// is halved. The structure is unchanged, so the move chosen for the downgraded board is played as is.
inline board_t downgrade(board_t b) {
    int present = 0, mt = 0;
    for (int i = 0; i < 16; i++) { int v = cell(b, i); present |= 1 << v; mt = std::max(mt, v); }
    if (mt < 15) return b;
    int k = mt - 1;
    while (k >= 1 && (present >> k & 1)) k--;
    if (k < 1) return b;
    board_t out = 0;
    for (int i = 0; i < 16; i++) { int v = cell(b, i); out |= board_t(v > k ? v - 1 : v) << (4 * i); }
    return out;
}
inline float leaf(board_t b) { return O.heur ? heur_value(b) : net.value(b); }

struct TTEntry { board_t key; float val; int8_t depth; };

// In goal mode a move that creates the goal tile is worth exactly 1 and every other reward is 0.
inline bool goal_hit(board_t af, int big) { return O.goal > 0 && (big || max_tile(af) >= O.goal); }
inline float reward(uint32_t r) { return O.goal > 0 && O.goal_bonus <= 0 ? 0.f : float(r); }
inline float goal_value(uint32_t r) { return O.goal_bonus > 0 ? float(r) + O.goal_bonus : 1.f; }

struct Searcher {
    std::vector<TTEntry> tt;
    uint64_t mask;
    long evals = 0;
    Searcher() : tt(size_t(1) << 22), mask((size_t(1) << 22) - 1) {
        for (auto& e : tt) e.key = ~0ULL, e.depth = -1;
    }
    static inline uint64_t h(board_t b) {
        b ^= b >> 33; b *= 0xff51afd7ed558ccdULL; b ^= b >> 33; b *= 0xc4ceb9fe1a85ec53ULL; b ^= b >> 33;
        return b;
    }
    float max_node(board_t s, int d, float p) {
        float best = 0;  // terminal (no legal move) is worth 0
        if (!O.heur && (d == 0 || p < O.cprob)) {  // all children are leaves: evaluate them in one batch
            board_t afs[4];
            float rs[4], vals[4];
            int n = 0;
            for (int a = 0; a < 4; a++) {
                uint32_t r; int big;
                board_t af = do_move(s, a, &r, &big);
                if (af == s) continue;
                if (goal_hit(af, big)) { best = std::max(best, goal_value(r)); continue; }
                if (big) { best = std::max(best, r + 1e6f); continue; }
                afs[n] = af; rs[n] = reward(r); n++;
            }
            net.value_batch(afs, n, vals);
            evals += n;
            for (int k = 0; k < n; k++) best = std::max(best, rs[k] + vals[k]);
            return best;
        }
        for (int a = 0; a < 4; a++) {
            uint32_t r; int big;
            board_t af = do_move(s, a, &r, &big);
            if (af == s) continue;
            if (O.heur) r = 0;
            float v = goal_hit(af, big) ? goal_value(r) : big ? r + 1e6f : reward(r) + chance(af, d, p);
            if (v > best) best = v;
        }
        return best;
    }
    float chance(board_t af, int d, float p) {
        if (d == 0 || p < O.cprob) { evals++; return leaf(af); }
        TTEntry& e = tt[h(af) & mask];
        if (e.key == af && e.depth >= d) return e.val;
        uint64_t m = empty_mask(af);
        int n = __builtin_popcountll(m);
        float sum = 0;
        float pc = p / n;
        while (m) {
            int pos = __builtin_ctzll(m);
            m &= m - 1;
            sum += 0.9f * max_node(af | (board_t(1) << pos), d - 1, pc * 0.9f);
            sum += 0.1f * max_node(af | (board_t(2) << pos), d - 1, pc * 0.1f);
        }
        float v = sum / n;
        e.key = af; e.val = v; e.depth = int8_t(d);
        return v;
    }
    int best_move(board_t s, int d) {
        int bestd = -1;
        float best = -1e30f;
        for (int a = 0; a < 4; a++) {
            uint32_t r; int big;
            board_t af = do_move(s, a, &r, &big);
            if (af == s) continue;
            if (O.heur) r = 0;
            float v = goal_hit(af, big) ? goal_value(r) : big ? r + 1e6f : reward(r) + chance(af, d, 1.0f);
            if (v > best) { best = v; bestd = a; }
        }
        return bestd;
    }
};

std::mutex mtx;
std::atomic<int> next_game{0};
struct Result { double score; int maxt; long moves; double secs; long corner; };
std::vector<Result> results;
FILE* dumpf = nullptr;
FILE* logf_ = nullptr;

void worker(int tid) {
    Searcher S;
    while (true) {
        int g = next_game.fetch_add(1);
        if (g >= O.games) break;
        Rng rng(uint64_t(O.seed) * 1000003ULL + g);
        auto t0 = std::chrono::steady_clock::now();
        board_t b = start_states.empty() ? initial_board(rng) : start_states[size_t(g) % start_states.size()];
        double score = 0;
        long moves = 0;
        int maxt = 0, stage = 0;
        long corner = 0;  // moves after which the largest tile sits in a corner
        if (O.trace) { printf("S %d %016llx\n", g, (unsigned long long)b); fflush(stdout); }
        while (true) {
            int d = O.depth;
            if (O.adaptive > 0 && count_empty(b) <= O.adaptive) d++;
            {
                int e = count_empty(b);
                for (auto& pr : O.deep) if (e <= pr.first) d = std::max(d, pr.second);
            }
            int a = S.best_move(O.downgrade ? downgrade(b) : b, d);
            if (a < 0) break;
            uint32_t r; int big;
            board_t af = do_move(b, a, &r, &big);
            score += r;
            moves++;
            if (O.trace) {
                board_t nb = big ? af : spawn(af, rng);  // spawn here so the trace shows the next state
                printf("M %d %016llx %016llx %.0f\n", a, (unsigned long long)af, (unsigned long long)nb, score);
                fflush(stdout);
                if (big) { maxt = 16; break; }
                maxt = std::max(maxt, max_tile(af));
                if (O.goal > 0 && maxt >= O.goal) break;
                b = nb;
                continue;
            }
            if (big) { maxt = 16; break; }
            maxt = std::max(maxt, max_tile(af));
            if (O.goal > 0 && maxt >= O.goal) break;
            {
                int m = max_tile(af);
                corner += cell(af, 0) == m || cell(af, 3) == m || cell(af, 12) == m || cell(af, 15) == m;
            }
            if (dumpf && !O.heur) {
                Network tmp; tmp.nstages = 3; tmp.stage_mode = O.dump_mode;
                int st = tmp.stage(af);
                if (st != stage) {
                    std::lock_guard<std::mutex> lk(mtx);
                    fwrite(&af, 8, 1, dumpf);
                    stage = st;
                }
            }
            b = spawn(af, rng);
        }
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (O.trace) { printf("E %d %.0f %d %ld\n", g, score, maxt ? 1 << maxt : 0, moves); fflush(stdout); }
        std::lock_guard<std::mutex> lk(mtx);
        results.push_back({score, maxt, moves, secs, corner});
        if (logf_) { fprintf(logf_, "%d,%.0f,%d,%ld,%.2f\n", g, score, maxt ? 1 << maxt : 0, moves, secs); fflush(logf_); }
        if (O.verbose)
            fprintf(stderr, "game %d score %.0f max %d moves %ld %.1fs (done %zu)\n", g, score, 1 << maxt, moves, secs, results.size());
    }
}

int main(int argc, char** argv) {
    init_tables();
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (k == "--weights") O.weights = nx();
        else if (k == "--games") O.games = std::stoi(nx());
        else if (k == "--threads") O.threads = std::stoi(nx());
        else if (k == "--depth") O.depth = std::stoi(nx());
        else if (k == "--adaptive") O.adaptive = std::stoi(nx());
        else if (k == "--cprob") O.cprob = std::stof(nx());
        else if (k == "--seed") O.seed = std::stoi(nx());
        else if (k == "--dump") O.dump = nx();
        else if (k == "--log") O.log = nx();
        else if (k == "-v") O.verbose = true;
        else if (k == "--heur") O.heur = true;
        else if (k == "--downgrade") O.downgrade = true;
        else if (k == "--dump-mode") O.dump_mode = std::stoi(nx());
        else if (k == "--starts") O.starts = nx();
        else if (k == "--trace") O.trace = true;
        else if (k == "--goal") O.goal = std::stoi(nx());
        else if (k == "--goal-bonus") O.goal_bonus = std::stof(nx());
        else if (k == "--deep") {  // e.g. 6:3,3:4,1:5
            std::string v = nx();
            size_t pos = 0;
            while (pos < v.size()) {
                size_t c = v.find(',', pos);
                std::string item = v.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
                size_t col = item.find(':');
                O.deep.push_back({std::stoi(item.substr(0, col)), std::stoi(item.substr(col + 1))});
                if (c == std::string::npos) break;
                pos = c + 1;
            }
        }
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 1; }
    }
    init_heur();
    if (O.heur) O.weights = "(heuristic)";
    else if (!net.load(O.weights)) { fprintf(stderr, "cannot load %s\n", O.weights.c_str()); return 1; }
    if (!O.starts.empty()) {
        FILE* f = fopen(O.starts.c_str(), "rb");
        board_t b;
        while (f && fread(&b, 8, 1, f) == 1) start_states.push_back(b);
        if (f) fclose(f);
        printf("%zu start states\n", start_states.size());
    }
    if (!O.dump.empty()) dumpf = fopen(O.dump.c_str(), "wb");
    if (!O.log.empty()) logf_ = fopen(O.log.c_str(), "w");
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> th;
    for (int i = 0; i < O.threads; i++) th.emplace_back(worker, i);
    for (auto& t : th) t.join();
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (dumpf) fclose(dumpf);
    if (logf_) fclose(logf_);

    int n = int(results.size());
    double sum = 0, mx = 0;
    long moves = 0, corner = 0;
    int reach[18] = {};
    for (auto& r : results) {
        sum += r.score; mx = std::max(mx, r.score); moves += r.moves; corner += r.corner;
        for (int k = 1; k <= r.maxt; k++) reach[k]++;
    }
    for (auto& pr : O.deep) printf("[deep: <=%d empty -> depth %d] ", pr.first, pr.second);
    printf("weights=%s depth=%d adaptive=%d cprob=%g games=%d  time %.0fs  (%.2f ms/move)\n", O.weights.c_str(), O.depth,
           O.adaptive, O.cprob, n, secs, 1000.0 * secs * O.threads / std::max(1L, moves));
    printf("avg score %.0f  max score %.0f  avg moves %.0f  max-tile-in-corner %.1f%% of moves\n", sum / n, mx,
           double(moves) / n, 100.0 * corner / std::max(1L, moves));
    for (int k = 11; k <= 16; k++) {
        double p = double(reach[k]) / n;
        double ci = 1.96 * std::sqrt(p * (1 - p) / n);
        printf("  reach %6d: %6.2f%% (+/- %.2f)\n", 1 << k, 100 * p, 100 * ci);
    }
}
