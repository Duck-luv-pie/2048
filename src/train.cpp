// TD(0) afterstate learning of an n-tuple network for 2048 (Hogwild multi-threaded,
// backward updates per episode). Supports optimistic initialisation (OTD) and
// temporal coherence (TC) adaptive learning rates.
#include "ntuple.hpp"
#include <atomic>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <thread>

struct Args {
    std::string net = "4x6", out = "weights/net.bin", load, starts;
    int stages = 1, mode = 0, threads = 14;
    long games = 1000000, tc_after = -1, report = 10000;
    float alpha = 0.1f, init = 0.f, tc_alpha = 1.0f, start_prob = 0.f;
    long save_every = 200000;
    float restart_prob = 0;
    int restart_per_ep = 4, restart_min_stage = 1;
    long restart_cap = 2000000;
    bool lock = false, drop_tc = false;
    int promote = 0, promote_mode = 1;
    float promote_init = -1, promote_bonus = 0, promote_unseen = 0;
    bool no_tc = false;
    long alpha_decay1 = -1, alpha_decay2 = -1;  // games after which alpha is divided by 10
};

static Network net;
static std::atomic<long> games_done{0};
static std::mutex stat_mtx;
struct Stats {
    long n = 0;
    double score = 0;
    long reach[18] = {};
    long moves = 0;
} stats;

static std::vector<board_t> start_states;

// Restart buffer: afterstates sampled from the second half of finished episodes. New episodes may
// restart from them (after a fresh random spawn), concentrating learning where games are lost.
static std::vector<board_t> restart_buf;
static std::atomic<uint64_t> restart_n{0};

void worker(int id, const Args& a) {
    Rng rng(0x1234567ULL * (id + 1) + uint64_t(time(nullptr)));
    std::vector<std::pair<board_t, uint32_t>> traj;
    traj.reserve(100000);
    while (true) {
        long g = games_done.fetch_add(1);
        if (g >= a.games) break;
        float alpha = net.tc ? a.tc_alpha : a.alpha;
        if (!net.tc && a.alpha_decay1 >= 0 && g >= a.alpha_decay1) alpha *= 0.1f;
        if (!net.tc && a.alpha_decay2 >= 0 && g >= a.alpha_decay2) alpha *= 0.1f;
        alpha /= net.num_features();
        board_t b;
        bool from_start = !start_states.empty() && rng.uniform() < a.start_prob;
        uint64_t nbuf = restart_n.load();
        bool from_buf = a.restart_prob > 0 && nbuf >= 10000 && rng.uniform() < a.restart_prob;
        if (from_buf)
            b = spawn(restart_buf[rng.below(uint32_t(std::min<uint64_t>(nbuf, restart_buf.size())))], rng);
        else if (from_start)
            b = start_states[rng.below(uint32_t(start_states.size()))];
        else
            b = initial_board(rng);
        int maxt = max_tile(b);
        traj.clear();
        double score = 0;
        bool big_end = false;
        while (true) {
            board_t best_after = 0;
            float best = -1e30f;
            uint32_t best_r = 0;
            int best_big = 0;
            board_t afs[4];
            uint32_t rs[4];
            int bigs[4], na = 0;
            for (int d = 0; d < 4; d++) {
                board_t af = do_move(b, d, &rs[na], &bigs[na]);
                if (af != b) afs[na++] = af;
            }
            float vals[4];
            net.value_batch(afs, na, vals);
            for (int k = 0; k < na; k++) {
                float v = rs[k] + vals[k];
                if (v > best) { best = v; best_after = afs[k]; best_r = rs[k]; best_big = bigs[k]; }
            }
            if (best == -1e30f) break;
            traj.push_back({best_after, best_r});
            score += best_r;
            if (best_big) { big_end = true; maxt = 16; break; }
            int mt = max_tile(best_after);
            if (mt > maxt) maxt = mt;
            b = spawn(best_after, rng);
        }
        // backward TD(0) updates
        float target = 0;
        int T = int(traj.size());
        Network::Feat FA, FB;
        Network::Feat *cur = &FA, *nxt = &FB;
        if (T > 0) net.features(traj[T - 1].first, *cur, net.tc);
        for (int t = T - 1; t >= 0; t--) {
            if (t > 0) net.features(traj[t - 1].first, *nxt, net.tc);  // software pipelining
            float v = net.sum(*cur);
            if (t == T - 1 && big_end) target = v;
            net.update(*cur, target - v, alpha);
            target = traj[t].second + net.sum(*cur);
            std::swap(cur, nxt);
        }
        if (a.restart_prob > 0 && T >= 8 && net.stage(traj[T - 1].first) >= a.restart_min_stage) {
            for (int k = 0; k < a.restart_per_ep; k++) {
                int t = T / 2 + int(rng.below(uint32_t(T - T / 2)));
                if (net.stage(traj[t].first) < a.restart_min_stage) continue;
                restart_buf[restart_n.fetch_add(1) % restart_buf.size()] = traj[t].first;
            }
        }
        // statistics over games from the initial position (or from start states, if every game uses one)
        if (!from_buf && (!from_start || a.start_prob >= 1)) {
            std::lock_guard<std::mutex> lk(stat_mtx);
            stats.n++;
            stats.score += score;
            stats.moves += T;
            for (int k = 1; k <= maxt; k++) stats.reach[k]++;
            if (stats.n >= a.report) {
                auto now = std::chrono::system_clock::now();
                std::time_t tt = std::chrono::system_clock::to_time_t(now);
                char buf[32];
                strftime(buf, sizeof buf, "%H:%M:%S", localtime(&tt));
                printf("[%s] games %ld avg %.0f moves/game %.0f | 2048 %.1f%% 4096 %.1f%% 8192 %.1f%% 16384 %.1f%% 32768 %.2f%% 65536 %.2f%%%s\n",
                       buf, g + 1, stats.score / stats.n, double(stats.moves) / stats.n,
                       100.0 * stats.reach[11] / stats.n, 100.0 * stats.reach[12] / stats.n,
                       100.0 * stats.reach[13] / stats.n, 100.0 * stats.reach[14] / stats.n,
                       100.0 * stats.reach[15] / stats.n, 100.0 * stats.reach[16] / stats.n,
                       net.tc ? " [TC]" : "");
                fflush(stdout);
                stats = Stats();
            }
        }
        if (a.save_every > 0 && g > 0 && g % a.save_every == 0) { net.save(a.out); net.save_tc(a.out + ".tc"); }
    }
}

int main(int argc, char** argv) {
    init_tables();
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (k == "--net") a.net = nx();
        else if (k == "--out") a.out = nx();
        else if (k == "--load") a.load = nx();
        else if (k == "--stages") a.stages = std::stoi(nx());
        else if (k == "--mode") a.mode = std::stoi(nx());
        else if (k == "--threads") a.threads = std::stoi(nx());
        else if (k == "--games") a.games = std::stol(nx());
        else if (k == "--tc-after") a.tc_after = std::stol(nx());
        else if (k == "--report") a.report = std::stol(nx());
        else if (k == "--alpha") a.alpha = std::stof(nx());
        else if (k == "--tc-alpha") a.tc_alpha = std::stof(nx());
        else if (k == "--init") a.init = std::stof(nx());
        else if (k == "--starts") a.starts = nx();
        else if (k == "--start-prob") a.start_prob = std::stof(nx());
        else if (k == "--save-every") a.save_every = std::stol(nx());
        else if (k == "--promote") a.promote = std::stoi(nx());
        else if (k == "--restart-prob") a.restart_prob = std::stof(nx());
        else if (k == "--lock") a.lock = true;
        else if (k == "--drop-tc") a.drop_tc = true;
        else if (k == "--restart-per-ep") a.restart_per_ep = std::stoi(nx());
        else if (k == "--restart-min-stage") a.restart_min_stage = std::stoi(nx());
        else if (k == "--promote-mode") a.promote_mode = std::stoi(nx());
        else if (k == "--promote-init") a.promote_init = std::stof(nx());
        else if (k == "--promote-bonus") a.promote_bonus = std::stof(nx());
        else if (k == "--promote-unseen") a.promote_unseen = std::stof(nx());
        else if (k == "--no-tc") a.no_tc = true;
        else if (k == "--decay1") a.alpha_decay1 = std::stol(nx());
        else if (k == "--decay2") a.alpha_decay2 = std::stol(nx());
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 1; }
    }
    if (!a.load.empty()) {
        if (!net.load(a.load)) { fprintf(stderr, "load failed\n"); return 1; }
        printf("loaded %s (%zu patterns, %d stages)\n", a.load.c_str(), net.pats.size(), net.nstages);
        if (!a.no_tc && net.load_tc(a.load + ".tc")) printf("loaded TC state\n");
        if (a.promote > net.nstages) {
            net.promote(a.promote, a.promote_mode, a.promote_init, a.promote_bonus, a.promote_unseen);
            printf("promoted to %d stages (mode %d)\n", net.nstages, net.stage_mode);
        }
        if (a.drop_tc) { net.disable_tc(); printf("TC disabled: plain TD\n"); }
    } else {
        net.init(a.net, a.stages, a.mode, a.init, false);
    }
    if (!a.starts.empty()) {
        FILE* f = fopen(a.starts.c_str(), "rb");
        board_t b;
        while (f && fread(&b, 8, 1, f) == 1) start_states.push_back(b);
        if (f) fclose(f);
        printf("loaded %zu start states\n", start_states.size());
    }
    restart_buf.assign(a.restart_cap, 0);
    if (a.lock) net.lock_memory();
    // Phase 1: plain TD (possibly optimistic). Phase 2: TC.
    long total = a.games;
    if (a.tc_after == 0) net.enable_tc();
    if (a.tc_after > 0 && a.tc_after < total) {
        a.games = a.tc_after;
        {
            std::vector<std::thread> th;
            for (int i = 0; i < a.threads; i++) th.emplace_back(worker, i, std::cref(a));
            for (auto& t : th) t.join();
        }
        net.enable_tc();
        printf("switching to TC learning\n");
        a.games = total;
    }
    std::vector<std::thread> th;
    for (int i = 0; i < a.threads; i++) th.emplace_back(worker, i, std::cref(a));
    for (auto& t : th) t.join();
    net.save(a.out);
    net.save_tc(a.out + ".tc");
    printf("saved %s\n", a.out.c_str());
}
