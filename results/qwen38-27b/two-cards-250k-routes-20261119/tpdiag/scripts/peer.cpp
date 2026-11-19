#include <level_zero/ze_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
constexpr size_t capacity = 64 * 1024 * 1024;
constexpr uint64_t wait_limit_ns = 1'000'000'000;
constexpr unsigned repeats = 25;
constexpr size_t chain_slot = 160 * 1024;  // 16 rows x 5,120 x FP16, TensorFold's widest verify exchange
constexpr unsigned chain_steps = 128;        // 64 layers x 2 reductions, one forward

void check(ze_result_t result, const char *operation) {
    if (result != ZE_RESULT_SUCCESS) {
        std::fprintf(stderr, "%s failed: %#x\n", operation, unsigned(result));
        throw std::runtime_error(operation);
    }
}

#define ZE(call) check((call), #call)

double elapsed(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

struct Stream {
    ze_command_list_handle_t list{};
    ze_command_queue_handle_t queue{};

    void init(ze_context_handle_t context, ze_device_handle_t device, unsigned ordinal,
              bool immediate) {
        ze_command_queue_desc_t desc{};
        desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        desc.ordinal = ordinal;
        desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
        // In-order, as TensorFold's lists (ze.zig immediate()): check_payload appends a kernel and a
        // copy with no event between them and relies on list order.
        if (immediate) {
            desc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
            ZE(zeCommandListCreateImmediate(context, device, &desc, &list));
        } else {
            ZE(zeCommandQueueCreate(context, device, &desc, &queue));
            ze_command_list_desc_t list_desc{};
            list_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
            list_desc.flags = ZE_COMMAND_LIST_FLAG_IN_ORDER;
            list_desc.commandQueueGroupOrdinal = ordinal;
            ZE(zeCommandListCreate(context, device, &list_desc, &list));
        }
    }

    void close() {
        if (queue) ZE(zeCommandListClose(list));
    }

    void submit() {
        if (queue) ZE(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    }

    void reset() {
        if (queue) {
            ZE(zeCommandQueueSynchronize(queue, wait_limit_ns));
            ZE(zeCommandListReset(list));
        }
    }
};

struct Rank {
    ze_device_handle_t device{};
    ze_device_properties_t props{};
    ze_module_handle_t module{};
    void *src{}, *inbox{}, *result{}, *flag{}, *error{};
    uint32_t *host{};
    Stream compute[2], copy[2];
    ze_event_handle_t aux{}, done{}, send{}, marker{}, seen{}, fin{};
    ze_kernel_handle_t scalar{}, vector{}, publish{}, await{}, reduce{}, verify{}, stamp{};
    ze_kernel_handle_t produce{}, reduce_check{}, await_ge{};
    void *ring{}, *ring_in{}, *out{};  // chain test: 4-slot send ring, peer inbox ring, result
    ze_command_list_handle_t chain_list{};  // chain test: regular in-order list for replay
    ze_command_queue_handle_t chain_queue{};
};

ze_kernel_handle_t kernel(ze_module_handle_t module, const char *name, unsigned work_group) {
    ze_kernel_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    desc.pKernelName = name;
    ze_kernel_handle_t k;
    ZE(zeKernelCreate(module, &desc, &k));
    ZE(zeKernelSetGroupSize(k, work_group, 1, 1));
    return k;
}

template<class... Args>
void launch(Stream &s, ze_kernel_handle_t k, unsigned groups, ze_event_handle_t done,
            ze_event_handle_t dependency, Args... args) {
    unsigned arg = 0;
    (ZE(zeKernelSetArgumentValue(k, arg++, sizeof(args), &args)), ...);
    ze_group_count_t sizes{groups, 1, 1};
    ZE(zeCommandListAppendLaunchKernel(s.list, k, &sizes, done, dependency ? 1 : 0,
                                      dependency ? &dependency : nullptr));
}

void event_wait(ze_event_handle_t ev) { ZE(zeEventHostSynchronize(ev, wait_limit_ns)); }
void event_reset(ze_event_handle_t ev) { ZE(zeEventHostReset(ev)); }

double event_us(const Rank &rank, ze_event_handle_t event) {
    ze_kernel_timestamp_result_t time{};
    auto rc = zeEventQueryKernelTimestamp(event, &time);
    if (rc == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE || rc == ZE_RESULT_ERROR_NOT_AVAILABLE ||
        rc == ZE_RESULT_ERROR_INVALID_ARGUMENT) return -1;
    ZE(rc);
    const unsigned bits = rank.props.kernelTimestampValidBits;
    if (bits == 0 || bits > 64 || rank.props.timerResolution == 0) return -1;
    uint64_t ticks = time.global.kernelEnd - time.global.kernelStart;
    if (bits < 64) ticks &= (uint64_t{1} << bits) - 1;
    return double(ticks) * rank.props.timerResolution / 1000;
}

void copy_bytes(Stream &stream, void *dst, const void *src, size_t bytes,
                ze_event_handle_t signal = nullptr) {
    ZE(zeCommandListAppendMemoryCopy(stream.list, dst, src, bytes, signal, 0, nullptr));
}

void fill(Rank &rank, void *buffer, uint32_t value, size_t bytes) {
    ZE(zeCommandListAppendMemoryFill(rank.compute[1].list, buffer, &value, sizeof(value), bytes,
                                   rank.aux, 0, nullptr));
    event_wait(rank.aux);
    event_reset(rank.aux);
}

unsigned groups_for(size_t words) {
    return unsigned(std::min((words + 63) / 64, size_t{2048}));
}

unsigned check_payload(Rank &rank, void *data, size_t bytes, uint32_t expected, bool check_flag) {
    auto &stream = rank.compute[1];
    fill(rank, rank.error, 0, sizeof(uint32_t));
    launch(stream, rank.verify, groups_for(bytes / 4), nullptr, nullptr, data,
           uint32_t(bytes / 4), expected, rank.error);
    copy_bytes(stream, rank.host, rank.error, sizeof(uint32_t),
               check_flag ? nullptr : rank.aux);
    if (check_flag) {
        copy_bytes(stream, rank.host + 1, static_cast<char *>(rank.error) + 4, 4, rank.aux);
    }
    event_wait(rank.aux);
    event_reset(rank.aux);
    return rank.host[0] + (check_flag ? rank.host[1] : 0);
}

void init_rank(Rank &r, ze_context_handle_t context, ze_event_pool_handle_t pool, size_t i,
               const std::vector<uint8_t> &image) {
    r.props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
    ZE(zeDeviceGetProperties(r.device, &r.props));
    unsigned count = 0;
    ZE(zeDeviceGetCommandQueueGroupProperties(r.device, &count, nullptr));
    std::vector<ze_command_queue_group_properties_t> queues(count);
    for (auto &q : queues) q.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    ZE(zeDeviceGetCommandQueueGroupProperties(r.device, &count, queues.data()));
    unsigned compute = count, copy = count;
    for (unsigned j = 0; j < count; ++j) {
        if ((queues[j].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) && compute == count) {
            compute = j;
        }
        if ((queues[j].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
            !(queues[j].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) && copy == count) {
            copy = j;
        }
    }
    if (compute == count || copy == count) throw std::runtime_error("missing compute or copy-only group");
    std::fprintf(stderr, "rank %zu command groups compute=%u copy_only=%u clock_tick_ns=%llu bits=%u\n",
                 i, compute, copy, (unsigned long long)r.props.timerResolution,
                 r.props.kernelTimestampValidBits);
    for (bool immediate : {false, true}) {
        r.compute[immediate].init(context, r.device, compute, immediate);
        r.copy[immediate].init(context, r.device, copy, immediate);
    }

    ze_device_mem_alloc_desc_t mem_desc{};
    mem_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    ZE(zeMemAllocDevice(context, &mem_desc, capacity, 4096, r.device, &r.src));
    ZE(zeMemAllocDevice(context, &mem_desc, capacity, 4096, r.device, &r.inbox));
    ZE(zeMemAllocDevice(context, &mem_desc, 40 * 1024, 4096, r.device, &r.result));
    ZE(zeMemAllocDevice(context, &mem_desc, 64, 64, r.device, &r.flag));
    ZE(zeMemAllocDevice(context, &mem_desc, 64, 64, r.device, &r.error));
    ze_host_mem_alloc_desc_t host_desc{};
    host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    void *host;
    ZE(zeMemAllocHost(context, &host_desc, 64, 64, &host));
    r.host = static_cast<uint32_t *>(host);

    ze_module_desc_t mod_desc{};
    mod_desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    mod_desc.format = ZE_MODULE_FORMAT_NATIVE;
    mod_desc.inputSize = image.size();
    mod_desc.pInputModule = image.data();
    ZE(zeModuleCreate(context, r.device, &mod_desc, &r.module, nullptr));
    r.scalar = kernel(r.module, "store4", 64);
    r.vector = kernel(r.module, "store16", 64);
    r.publish = kernel(r.module, "publish", 1);
    r.await = kernel(r.module, "await_epoch", 1);
    r.reduce = kernel(r.module, "reduce_pair", 64);
    r.verify = kernel(r.module, "check_words", 64);
    r.stamp = kernel(r.module, "stamp", 1);

    ze_event_desc_t event_desc{};
    event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    event_desc.index = unsigned(i * 5);
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    for (auto *event : {&r.aux, &r.done, &r.send, &r.marker, &r.seen}) {
        ZE(zeEventCreate(pool, &event_desc, event));
        ++event_desc.index;
    }
    event_desc.index = unsigned(10 + i);
    ZE(zeEventCreate(pool, &event_desc, &r.fin));
    r.produce = kernel(r.module, "produce", 64);
    r.reduce_check = kernel(r.module, "reduce_check", 64);
    r.await_ge = kernel(r.module, "await_ge", 1);
    ZE(zeMemAllocDevice(context, &mem_desc, 4 * chain_slot, 4096, r.device, &r.ring));
    ZE(zeMemAllocDevice(context, &mem_desc, 4 * chain_slot, 4096, r.device, &r.ring_in));
    ZE(zeMemAllocDevice(context, &mem_desc, chain_slot, 4096, r.device, &r.out));
    {
        ze_command_queue_desc_t qd{};
        qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        qd.ordinal = compute;
        qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
        ZE(zeCommandQueueCreate(context, r.device, &qd, &r.chain_queue));
        ze_command_list_desc_t ld{};
        ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
        ld.commandQueueGroupOrdinal = compute;
        ld.flags = ZE_COMMAND_LIST_FLAG_IN_ORDER;
        ZE(zeCommandListCreate(context, r.device, &ld, &r.chain_list));
    }
    fill(r, r.flag, 0, 4);
}

struct Arm {
    const char *name;
    bool copy;
    bool immediate;
    unsigned width;
};

constexpr Arm arms[] = {
    {"copy_regular", true, false, 0}, {"copy_immediate", true, true, 0},
    {"store4_regular", false, false, 4}, {"store4_immediate", false, true, 4},
    {"store16_regular", false, false, 16}, {"store16_immediate", false, true, 16}
};

void send(Rank &from, Rank &to, const Arm &arm, size_t bytes) {
    if (arm.copy) {
        copy_bytes(from.copy[arm.immediate], to.inbox, from.src, bytes, from.send);
    } else {
        const auto entries = uint32_t(bytes / arm.width);
        launch(from.compute[arm.immediate], arm.width == 4 ? from.scalar : from.vector,
               groups_for(entries), from.send, nullptr, from.src, to.inbox, entries);
    }
}

void log_sample(const char *test, const Arm &arm, const char *direction, size_t bytes,
                unsigned rep, double append, double submit, double wait, double wall,
                double gpu0, double gpu1, double flag0, double flag1, unsigned errors) {
    std::printf("%s,%s,%s,%zu,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%u\n",
                test, arm.name, direction, bytes, rep, append, submit, wait, wall,
                gpu0, gpu1, flag0, flag1, errors);
}

unsigned one_way(Rank &from, Rank &to, const Arm &arm, size_t bytes, unsigned rep,
                 const char *direction) {
    auto &s = arm.copy ? from.copy[arm.immediate] : from.compute[arm.immediate];
    uint32_t value = 0x12340601u + rep;
    fill(from, from.src, value, bytes);
    fill(to, to.inbox, 0, bytes);
    const auto a = Clock::now();
    send(from, to, arm, bytes);
    const auto b = Clock::now();
    s.close();
    const auto c = Clock::now();
    s.submit();
    const auto d = Clock::now();
    event_wait(from.send);
    const auto e = Clock::now();
    const auto gpu = event_us(from, from.send);
    s.reset();
    event_reset(from.send);
    const unsigned errors = check_payload(to, to.inbox, bytes, value, false);
    log_sample("peer", arm, direction, bytes, rep, elapsed(a, b), elapsed(c, d), elapsed(d, e),
               elapsed(a, e), gpu, -1, -1, -1, errors);
    return errors;
}

unsigned allreduce(std::array<Rank, 2> &ranks, const Arm &arm, size_t bytes, unsigned rep,
                   uint32_t epoch) {
    for (unsigned i = 0; i < 2; ++i) {
        auto &r = ranks[i];
        fill(r, r.src, std::bit_cast<uint32_t>(float(2 * rep + i + 1)), bytes);
        fill(r, r.inbox, 0, bytes);
        fill(r, r.result, 0, bytes);
        fill(r, r.error, 0xFFFFFFFF, 8);
    }
    const auto a = Clock::now();
    for (unsigned i = 0; i < 2; ++i) send(ranks[i], ranks[1 - i], arm, bytes);
    if (!arm.copy) {
        for (unsigned i = 0; i < 2; ++i) {
            auto &r = ranks[i];
            launch(r.compute[arm.immediate], r.publish, 1, r.marker, r.send,
                   ranks[1 - i].flag, epoch);
        }
    }
    for (unsigned i = 0; i < 2; ++i) {
        auto &r = ranks[i];
        auto &s = r.compute[arm.immediate];
        if (!arm.copy) {
            launch(s, r.await, 1, r.seen, r.marker, r.flag, epoch,
                   static_cast<char *>(r.error) + 4);
        }
        launch(s, r.reduce, groups_for(bytes / 4), r.done,
               arm.copy ? ranks[1 - i].send : r.seen, r.src, r.inbox, r.result, uint32_t(bytes / 4));
    }
    const auto b = Clock::now();
    for (auto &r : ranks) {
        r.compute[arm.immediate].close();
        if (arm.copy) r.copy[arm.immediate].close();
    }
    const auto c = Clock::now();
    if (arm.copy) for (auto &r : ranks) r.copy[arm.immediate].submit();
    for (auto &r : ranks) r.compute[arm.immediate].submit();
    const auto d = Clock::now();
    for (auto &r : ranks) event_wait(r.done);
    const auto e = Clock::now();
    std::array<double, 2> send_times{}, flag_times{};
    unsigned errors = 0;
    for (unsigned i = 0; i < 2; ++i) {
        auto &r = ranks[i];
        send_times[i] = event_us(r, r.send);
        flag_times[i] = arm.copy ? -1 : event_us(r, r.seen);
        r.compute[arm.immediate].reset();
        if (arm.copy) r.copy[arm.immediate].reset();
        for (auto ev : {r.done, r.send}) event_reset(ev);
        if (!arm.copy) for (auto ev : {r.marker, r.seen}) event_reset(ev);
        errors += check_payload(r, r.result, bytes, std::bit_cast<uint32_t>(float(4 * rep + 3)),
                                !arm.copy);
    }
    log_sample("allreduce", arm, "both", bytes, rep, elapsed(a, b), elapsed(c, d), elapsed(d, e),
               elapsed(a, e), send_times[0], send_times[1], flag_times[0], flag_times[1], errors);
    return errors;
}

unsigned host_probe(Rank &rank, const Arm &arm, unsigned rep, const char *direction) {
    auto &s = rank.compute[arm.immediate];
    fill(rank, rank.result, 0, 4);
    const auto a = Clock::now();
    launch(s, rank.stamp, 1, rank.done, nullptr, rank.result, uint32_t(rep + 1));
    const auto b = Clock::now();
    s.close();
    const auto c = Clock::now();
    s.submit();
    const auto d = Clock::now();
    event_wait(rank.done);
    const auto e = Clock::now();
    event_wait(rank.done);
    const auto f = Clock::now();
    const auto gpu = event_us(rank, rank.done);
    s.reset();
    event_reset(rank.done);
    const auto errors = check_payload(rank, rank.result, 4, rep + 1, false);
    log_sample("host_kernel", arm, direction, 4, rep, elapsed(a, b), elapsed(c, d), elapsed(d, e),
               elapsed(a, e), gpu, -1, elapsed(e, f), -1, errors);
    return errors;
}


// ---- chain test: one forward's worth (128) of produce -> exchange -> reduce steps per rank ----
void *at(void *p, size_t bytes) { return static_cast<char *>(p) + bytes; }

enum class ChainArm { local, event_only, event_read, store_flag };
struct ChainCfg { const char *name; ChainArm arm; unsigned events; };  // events: 0 none, 1 TF, 2 devpool, 3 host scope

struct ChainEvents {
    ze_event_pool_handle_t pool{};
    std::vector<ze_event_handle_t> ready;  // [step * 2 + rank]
    ze_context_handle_t ctx{};
    std::array<ze_device_handle_t, 2> devs{};
    bool hv{};
    ze_event_scope_flags_t sc{};
    // 04:00 fix: the first run reused host-reset events on the next rep and the next append returned
    // 0x78000004. The reset is still timed (TensorFold's cost); then the pool is rebuilt untimed.
    void renew() {
        for (auto e : ready) ZE(zeEventDestroy(e));
        ZE(zeEventPoolDestroy(pool));
        ready.clear();
        init(ctx, devs, hv, sc);
    }
    void init(ze_context_handle_t context, std::array<ze_device_handle_t, 2> devices, bool host_visible,
              ze_event_scope_flags_t scope) {
        ctx = context; devs = devices; hv = host_visible; sc = scope;
        ze_event_pool_desc_t pd{};
        pd.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
        pd.flags = host_visible ? ZE_EVENT_POOL_FLAG_HOST_VISIBLE : 0;
        pd.count = 2 * chain_steps;
        ZE(zeEventPoolCreate(context, &pd, 2, devices.data(), &pool));
        ready.resize(2 * chain_steps);
        for (unsigned k = 0; k < ready.size(); ++k) {
            ze_event_desc_t ed{};
            ed.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
            ed.index = k;
            ed.signal = scope;
            ed.wait = scope;
            ZE(zeEventCreate(pool, &ed, &ready[k]));
        }
    }
};

template<class... Args>
void launch_list(ze_command_list_handle_t list, ze_kernel_handle_t k, unsigned groups, ze_event_handle_t signal,
                 unsigned nwait, ze_event_handle_t *waits, Args... args) {
    unsigned arg = 0;
    (ZE(zeKernelSetArgumentValue(k, arg++, sizeof(args), &args)), ...);
    ze_group_count_t sizes{groups, 1, 1};
    ZE(zeCommandListAppendLaunchKernel(list, k, &sizes, signal, nwait, waits));
}

void chain_step(Rank &r, unsigned i, Rank &peer, ze_command_list_handle_t list, const ChainCfg &cfg,
                ChainEvents *ev, size_t bytes, unsigned s, uint32_t epoch) {
    const size_t slot = (s % 4) * chain_slot;
    const auto quads = uint32_t(bytes / 16);
    const unsigned groups = groups_for(quads);
    const float mine = float(i + 1) + 4.0f * float(s);
    const float both = 3.0f + 8.0f * float(s);
    void *send = at(r.ring, slot);
    void *mismatches = at(r.error, 8);
    void *timeouts = at(r.error, 12);
    ze_event_handle_t last = s + 1 == chain_steps ? r.fin : nullptr;
    launch_list(list, r.produce, groups, nullptr, 0, nullptr, send, quads, mine);
    switch (cfg.arm) {
    case ChainArm::local:
        launch_list(list, r.reduce_check, groups, last, 0, nullptr, send, send, r.out, quads, 2 * mine, mismatches);
        break;
    case ChainArm::event_only:
    case ChainArm::event_read: {
        ZE(zeCommandListAppendSignalEvent(list, ev->ready[s * 2 + i]));
        ze_event_handle_t waits[2] = {ev->ready[s * 2], ev->ready[s * 2 + 1]};
        void *other = cfg.arm == ChainArm::event_read ? at(peer.ring, slot) : send;
        launch_list(list, r.reduce_check, groups, last, 2, waits, send, other, r.out, quads,
                    cfg.arm == ChainArm::event_read ? both : 2 * mine, mismatches);
        break;
    }
    case ChainArm::store_flag:
        launch_list(list, r.produce, groups, nullptr, 0, nullptr, at(peer.ring_in, slot), quads, mine);
        launch_list(list, r.publish, 1, nullptr, 0, nullptr, peer.flag, epoch);
        launch_list(list, r.await_ge, 1, nullptr, 0, nullptr, r.flag, epoch, timeouts);
        launch_list(list, r.reduce_check, groups, last, 0, nullptr, send, at(r.ring_in, slot), r.out, quads,
                    both, mismatches);
        break;
    }
}

unsigned chain_test(std::array<Rank, 2> &ranks, ze_context_handle_t context,
                    std::array<ze_device_handle_t, 2> devices) {
    std::array<ChainEvents, 4> pools{};
    pools[1].init(context, devices, true, ZE_EVENT_SCOPE_FLAG_SUBDEVICE);   // tp.zig sharedEvents: host-visible pool, scope 1
    pools[2].init(context, devices, false, ZE_EVENT_SCOPE_FLAG_SUBDEVICE);  // same, device-only pool
    pools[3].init(context, devices, true, ZE_EVENT_SCOPE_FLAG_HOST);        // same, host scope (full flush)
    const ChainCfg cfgs[] = {
        {"local", ChainArm::local, 0}, {"store_flag", ChainArm::store_flag, 0},
        {"event_only", ChainArm::event_only, 1}, {"event_read", ChainArm::event_read, 1},
        {"event_read_devpool", ChainArm::event_read, 2}, {"event_read_hostscope", ChainArm::event_read, 3},
    };
    static uint32_t epoch = 1u << 20;
    unsigned errors = 0;
    for (bool replay : {false, true}) {  // every immediate cell first: a replay failure cannot cost them
        for (size_t bytes : {size_t{10240}, size_t{40960}, chain_slot}) {
            for (const auto &cfg : cfgs) {
                ChainEvents *ev = cfg.events ? &pools[cfg.events] : nullptr;
                double build = 0;
                if (replay) {
                    const auto a = Clock::now();
                    for (unsigned i = 0; i < 2; ++i) ZE(zeCommandListReset(ranks[i].chain_list));
                    for (unsigned s = 0; s < chain_steps; ++s)
                        for (unsigned i = 0; i < 2; ++i)
                            chain_step(ranks[i], i, ranks[1 - i], ranks[i].chain_list, cfg, ev, bytes, s, epoch + s);
                    for (unsigned i = 0; i < 2; ++i) ZE(zeCommandListClose(ranks[i].chain_list));
                    build = elapsed(a, Clock::now());
                }
                for (unsigned rep = 0; rep <= repeats; ++rep) try {
                    // A replayed list carries the epochs it was built with: restart the flags below them.
                    for (auto &r : ranks) {
                        fill(r, at(r.error, 8), 0, 8);
                        if (replay) fill(r, r.flag, 0, 4);
                    }
                    const auto a = Clock::now();
                    if (replay) {
                        for (auto &r : ranks) ZE(zeCommandQueueExecuteCommandLists(r.chain_queue, 1, &r.chain_list, nullptr));
                    } else {
                        for (unsigned s = 0; s < chain_steps; ++s)
                            for (unsigned i = 0; i < 2; ++i)
                                chain_step(ranks[i], i, ranks[1 - i], ranks[i].compute[1].list, cfg, ev, bytes, s,
                                           epoch + s);
                    }
                    const auto b = Clock::now();
                    for (auto &r : ranks) event_wait(r.fin);
                    const auto c = Clock::now();
                    // As tp.zig syncAll: host-synchronize both lists, then host-reset the events (timed).
                    for (auto &r : ranks) {
                        if (replay) ZE(zeCommandQueueSynchronize(r.chain_queue, wait_limit_ns));
                        else ZE(zeCommandListHostSynchronize(r.compute[1].list, wait_limit_ns));
                    }
                    const auto c2 = Clock::now();
                    if (ev) for (auto e : ev->ready) event_reset(e);
                    const auto d = Clock::now();
                    for (auto &r : ranks) event_reset(r.fin);
                    if (!replay) epoch += chain_steps;
                    if (ev && !replay) ev->renew();  // a replayed list holds these handles: keep them there
                    unsigned bad = 0, late = 0;
                    for (auto &r : ranks) {
                        ZE(zeCommandListAppendMemoryCopy(r.compute[1].list, r.host + 2, at(r.error, 8), 8, r.aux, 0, nullptr));
                        event_wait(r.aux);
                        event_reset(r.aux);
                        bad += r.host[2];
                        late += r.host[3];
                    }
                    errors += bad + late;
                    std::printf("chain,%s_%s,both,%zu,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%u,%u,%u\n",
                                cfg.name, replay ? "replay" : "immediate", bytes, rep,
                                replay ? build / chain_steps : elapsed(a, b) / chain_steps,
                                replay ? elapsed(a, b) : 0.0, elapsed(b, c), elapsed(a, c),
                                elapsed(a, c) / chain_steps, ev ? elapsed(c2, d) : 0.0, bad, late, bad + late);
                } catch (...) {
                    std::fprintf(stderr, "chain failed: arm %s %s bytes %zu rep %u\n", cfg.name,
                                 replay ? "replay" : "immediate", bytes, rep);
                    throw;
                }
                if (replay) epoch += chain_steps;
            }
        }
    }
    return errors;
}

int main(int argc, char **argv) try {
    if (argc != 3) throw std::runtime_error("usage: tpdiag-ze native-image peer|allreduce|host|chain|smoke");
    const std::string test = argv[2];
    if (test != "peer" && test != "allreduce" && test != "host" && test != "smoke" && test != "chain") {
        throw std::runtime_error("unknown test");
    }
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::ifstream file(argv[1], std::ios::binary);
    const std::vector<uint8_t> image{std::istreambuf_iterator<char>(file), {}};
    if (image.empty()) throw std::runtime_error("missing native module");

    ZE(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    unsigned count = 0;
    ZE(zeDriverGet(&count, nullptr));
    std::vector<ze_driver_handle_t> drivers(count);
    ZE(zeDriverGet(&count, drivers.data()));
    if (drivers.size() != 1) throw std::runtime_error("expected one Level Zero GPU driver");
    ze_driver_properties_t driver_props{};
    driver_props.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
    ZE(zeDriverGetProperties(drivers[0], &driver_props));
    std::fprintf(stderr, "driverVersionRaw=%u\n", driver_props.driverVersion);
    count = 0;  // fix 03:50: count still held the driver count; zeDeviceGet(count=1, nullptr) returned 0x78000005
    ZE(zeDeviceGet(drivers[0], &count, nullptr));
    if (count != 2) throw std::runtime_error("expected two Level Zero devices");
    std::array<ze_device_handle_t, 2> devices{};
    ZE(zeDeviceGet(drivers[0], &count, devices.data()));
    for (unsigned i = 0; i < 2; ++i) {
        ze_pci_ext_properties_t pci{};
        pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
        ZE(zeDevicePciGetPropertiesExt(devices[i], &pci));
        const auto &bdf = pci.address;
        ze_device_p2p_properties_t peer{};
        peer.stype = ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES;
        ZE(zeDeviceGetP2PProperties(devices[i], devices[1 - i], &peer));
        std::fprintf(stderr, "rank %u pci=%04x:%02x:%02x.%x access=%d atomics=%d\n",
                     i, bdf.domain, bdf.bus, bdf.device, bdf.function,
                     !!(peer.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS),
                     !!(peer.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS));
        if (!(peer.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS)) throw std::runtime_error("no P2P access");
    }
    ze_context_desc_t context_desc{};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ze_context_handle_t context;
    ZE(zeContextCreate(drivers[0], &context_desc, &context));
    ze_event_pool_desc_t pool_desc{};
    pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    pool_desc.count = 12;
    ze_event_pool_handle_t pool;
    ZE(zeEventPoolCreate(context, &pool_desc, count, devices.data(), &pool));
    std::array<Rank, 2> ranks{};
    for (size_t i = 0; i < ranks.size(); ++i) {
        ranks[i].device = devices[i];
        init_rank(ranks[i], context, pool, i, image);
    }
    for (unsigned i = 0; i < 2; ++i) {
        auto &peer = ranks[1 - i];
        ZE(zeContextMakeMemoryResident(context, ranks[i].device, peer.inbox, capacity));
        ZE(zeContextMakeMemoryResident(context, ranks[i].device, peer.flag, 64));
        ZE(zeContextMakeMemoryResident(context, ranks[i].device, peer.ring, 4 * chain_slot));
        ZE(zeContextMakeMemoryResident(context, ranks[i].device, peer.ring_in, 4 * chain_slot));
    }

    std::puts("test,arm,direction,bytes,rep,append_us,execute_us,event_wait_us,wall_us,gpu0_us,gpu1_us,flag0_us,flag1_us,errors");
    unsigned errors = 0;
    const unsigned last = test == "smoke" ? 1 : repeats;
    if (test == "host" || test == "smoke") {
        for (unsigned rep = 0; rep <= last; ++rep) {
            for (unsigned i = 0; i < 2; ++i) {
                for (const Arm &arm : {arms[2], arms[3]}) {
                    errors += host_probe(ranks[i], arm, rep, i == 0 ? "rank0" : "rank1");
                }
            }
        }
    }
    if (test == "peer" || test == "smoke") {
        const auto sizes = test == "smoke" ? std::vector<size_t>{64, 10240} :
            std::vector<size_t>{64, 256, 1024, 4096, 5120, 10240, 20480, 40960, 163840,
                                1048576, 4194304, 16777216, capacity};
        for (size_t size : sizes) for (unsigned rep = 0; rep <= last; ++rep) {
            for (size_t i = 0; i < 6; ++i) {
                const auto &arm = arms[(i + rep) % 6];
                for (unsigned x = 0; x < 2; ++x) {
                    const unsigned rank = (x + rep) % 2;
                    errors += one_way(ranks[rank], ranks[1 - rank], arm, size, rep,
                                      rank == 0 ? "0>1" : "1>0");
                }
            }
        }
    }
    if (test == "allreduce" || test == "smoke") {
        uint32_t epoch = 1;
        for (size_t size : {5120, 10240, 20480, 40960}) {
            for (unsigned rep = 0; rep <= last; ++rep) for (size_t i = 0; i < 6; ++i) {
                errors += allreduce(ranks, arms[(i + rep) % 6], size, rep, epoch++);
            }
        }
    }
    if (test == "chain" || test == "smoke") errors += chain_test(ranks, context, devices);
    std::fprintf(stderr, "mismatch or wait error total=%u\n", errors);
    return errors ? 10 : 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "stopped: %s\n", e.what());
    return 1;
}
