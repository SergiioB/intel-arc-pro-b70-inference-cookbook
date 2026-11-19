// Standalone reproducer: reused, host-reset timestamped events + cross-device waits on in-order immediate lists.
// Pattern of TensorFold's XPU_PROFILE run with --tp 2 (no TensorFold code). Usage: repro NATIVE_BIN NDEV [BATCHES STEPS]
// Every kernel signals a timestamped event from its device's own pool. Each step: bump on each device, each device
// signals a shared "ready" event, then sum2 on each device waits on both ready events. Batch end: device 0 waits for
// device 1's last event, then a blocking copy into host stack memory, host sync of both lists, host reset of every
// event, next batch. A 30 s alarm reports a hang (exit 2).
#include <level_zero/ze_api.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <unistd.h>
#include <vector>

static void check(ze_result_t r, const char *what, int line) {
    if (r != ZE_RESULT_SUCCESS) { std::fprintf(stderr, "%s failed: %#x (line %d)\n", what, unsigned(r), line); std::exit(1); }
}
#define ZE(x) check((x), #x, __LINE__)
static volatile int g_batch = -1, g_step = -1;
static void on_alarm(int) {
    char msg[128];
    int n = std::snprintf(msg, sizeof msg, "HANG: no progress for 30 s in batch %d step %d\n", g_batch, g_step);
    (void)!write(2, msg, n);
    _exit(2);
}

int main(int argc, char **argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: repro NATIVE_BIN NDEV [BATCHES STEPS]\n"); return 64; }
    const unsigned ndev = std::atoi(argv[2]);
    const int batches = argc > 3 ? std::atoi(argv[3]) : 3, steps = argc > 4 ? std::atoi(argv[4]) : 128;
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> bin{std::istreambuf_iterator<char>(f), {}};
    std::signal(SIGALRM, on_alarm);
    ZE(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    uint32_t n = 0;
    ZE(zeDriverGet(&n, nullptr));
    std::vector<ze_driver_handle_t> drv(n);
    ZE(zeDriverGet(&n, drv.data()));
    uint32_t nd = 0;
    ZE(zeDeviceGet(drv[0], &nd, nullptr));
    if (nd < ndev) { std::fprintf(stderr, "need %u devices, have %u\n", ndev, nd); return 1; }
    std::vector<ze_device_handle_t> dev(nd);
    ZE(zeDeviceGet(drv[0], &nd, dev.data()));
    dev.resize(ndev);
    ze_driver_properties_t dp{ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES};
    ZE(zeDriverGetProperties(drv[0], &dp));
    std::printf("driverVersion %u, %u device(s)\n", dp.driverVersion, ndev);
    ze_context_desc_t cd{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    ze_context_handle_t ctx;
    ZE(zeContextCreate(drv[0], &cd, &ctx));

    const int per_dev_events = 2 * steps + 1;
    std::vector<ze_command_list_handle_t> list(ndev);
    std::vector<ze_kernel_handle_t> bump(ndev), sum2(ndev);
    std::vector<int *> buf(ndev);
    std::vector<std::vector<ze_event_handle_t>> prof(ndev);
    for (unsigned d = 0; d < ndev; ++d) {
        ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
        qd.ordinal = 0; qd.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER; qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
        ZE(zeCommandListCreateImmediate(ctx, dev[d], &qd, &list[d]));
        ze_module_desc_t md{ZE_STRUCTURE_TYPE_MODULE_DESC};
        md.format = ZE_MODULE_FORMAT_NATIVE; md.inputSize = bin.size(); md.pInputModule = bin.data();
        ze_module_handle_t mod;
        ZE(zeModuleCreate(ctx, dev[d], &md, &mod, nullptr));
        ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC};
        kd.pKernelName = "bump"; ZE(zeKernelCreate(mod, &kd, &bump[d]));
        kd.pKernelName = "sum2"; ZE(zeKernelCreate(mod, &kd, &sum2[d]));
        ZE(zeKernelSetGroupSize(bump[d], 16, 1, 1));
        ZE(zeKernelSetGroupSize(sum2[d], 16, 1, 1));
        ze_device_mem_alloc_desc_t ad{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
        void *p; ZE(zeMemAllocDevice(ctx, &ad, 4096, 64, dev[d], &p)); buf[d] = static_cast<int *>(p);
        // The profiler pool: one per device, host visible, kernel timestamps (as TensorFold's Gpu.events()).
        ze_event_pool_desc_t pd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
        pd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP; pd.count = per_dev_events;
        ze_event_pool_handle_t pool; ZE(zeEventPoolCreate(ctx, &pd, 1, &dev[d], &pool));
        for (int i = 0; i < per_dev_events; ++i) {
            ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC}; ed.index = i;
            ze_event_handle_t e; ZE(zeEventCreate(pool, &ed, &e)); prof[d].push_back(e);
        }
    }
    for (unsigned a = 0; a < ndev; ++a) for (unsigned b = 0; b < ndev; ++b) if (a != b) ZE(zeContextMakeMemoryResident(ctx, dev[a], buf[b], 4096));
    // Shared cross-device events, as tp.zig sharedEvents: host-visible pool over all devices, signal/wait scope 1.
    ze_event_pool_desc_t spd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
    spd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE; spd.count = steps * ndev + ndev;
    ze_event_pool_handle_t spool; ZE(zeEventPoolCreate(ctx, &spd, ndev, dev.data(), &spool));
    std::vector<ze_event_handle_t> ready(steps * ndev + ndev);
    for (unsigned i = 0; i < ready.size(); ++i) {
        ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC}; ed.index = i; ed.signal = 1; ed.wait = 1;
        ZE(zeEventCreate(spool, &ed, &ready[i]));
    }
    for (int bt = 0; bt < batches; ++bt) {
        g_batch = bt;
        alarm(30);
        std::vector<int> used(ndev, 0);
        for (int s = 0; s < steps; ++s) {
            g_step = s;
            for (unsigned d = 0; d < ndev; ++d) {
                int v = int(d) + 1;
                ZE(zeKernelSetArgumentValue(bump[d], 0, sizeof(int *), &buf[d]));
                ZE(zeKernelSetArgumentValue(bump[d], 1, sizeof(int), &v));
                ze_group_count_t g{1, 1, 1};
                ZE(zeCommandListAppendLaunchKernel(list[d], bump[d], &g, prof[d][used[d]++], 0, nullptr));
                if (ndev > 1) ZE(zeCommandListAppendSignalEvent(list[d], ready[s * ndev + d]));
            }
            for (unsigned d = 0; d < ndev; ++d) {
                int *peer = buf[(d + 1) % ndev];
                ZE(zeKernelSetArgumentValue(sum2[d], 0, sizeof(int *), &buf[d]));
                ZE(zeKernelSetArgumentValue(sum2[d], 1, sizeof(int *), &peer));
                int *out = buf[d] + 16;
                ZE(zeKernelSetArgumentValue(sum2[d], 2, sizeof(int *), &out));
                ze_group_count_t g{1, 1, 1};
                ZE(zeCommandListAppendLaunchKernel(list[d], sum2[d], &g, prof[d][used[d]++], ndev > 1 ? ndev : 0,
                                                   ndev > 1 ? &ready[s * ndev] : nullptr));
            }
        }
        // Batch end, as forward.zig: device 1 signals, device 0 waits on it, then a blocking copy to host stack memory.
        for (unsigned d = 1; d < ndev; ++d) ZE(zeCommandListAppendSignalEvent(list[d], ready[steps * ndev + d]));
        if (ndev > 1) ZE(zeCommandListAppendWaitOnEvents(list[0], ndev - 1, &ready[steps * ndev + 1]));
        int host[4] = {0, 0, 0, 0};
        ZE(zeCommandListAppendMemoryCopy(list[0], host, buf[0] + 16, sizeof(int), nullptr, 0, nullptr));
        for (unsigned d = 0; d < ndev; ++d) ZE(zeCommandListHostSynchronize(list[d], UINT64_MAX));
        alarm(0);
        uint64_t ticks = 0;
        for (unsigned d = 0; d < ndev; ++d) for (int i = 0; i < used[d]; ++i) {
            ze_kernel_timestamp_result_t t{};
            ZE(zeEventQueryKernelTimestamp(prof[d][i], &t));
            ticks += t.global.kernelEnd - t.global.kernelStart;
            ZE(zeEventHostReset(prof[d][i]));
        }
        for (auto e : ready) ZE(zeEventHostReset(e));
        std::printf("batch %d done: out=%d, %llu timestamp ticks\n", bt, host[0], (unsigned long long)ticks);
        std::fflush(stdout);
    }
    std::printf("PASS: %d batches x %d steps on %u device(s)\n", batches, steps, ndev);
    return 0;
}
