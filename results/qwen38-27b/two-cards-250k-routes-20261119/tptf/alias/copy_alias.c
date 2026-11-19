// Level Zero copy checks for one large device allocation (no TensorFold code).
//   copy_alias straddle SIZE_MIB...  one copy of N bytes whose destination straddles base + 2^31 (and + 2^32 when the
//                                   buffer is big enough), whole and split at the boundary, immediate and regular lists;
//                                   checks offset 0 .. 4 MiB and the copied range.
//   copy_alias layout AT            TensorFold's prompt-cache save at AT tokens (Qwen3.8-27B, TP2 rank, int8 K/V):
//                                   64 layers' copies, in order, into one buffer; checks every word of it.
// Device: the first Level Zero device (ZE_AFFINITY_MASK picks it; unset = card 0 of a two-device context).
#include <level_zero/ze_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { ze_result_t r_ = (x); if (r_) { fprintf(stderr, "%s:%d %s = 0x%x\n", __FILE__, __LINE__, #x, r_); exit(2); } } while (0)
#define POISON 0xDEADBEEFu
#define CHUNK (64ull << 20)

static ze_context_handle_t ctx;
static ze_device_handle_t dev;
static ze_command_list_handle_t imm, reg;
static ze_command_queue_handle_t queue;
static int regular; // 1: copies go through a regular list and the queue

static void *dalloc(size_t bytes) {
    void *p = NULL;
    ze_relaxed_allocation_limits_exp_desc_t relaxed = {ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC, NULL,
                                                      ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE};
    ze_device_mem_alloc_desc_t dd = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, bytes >= (4ull << 30) ? &relaxed : NULL, 0, 0};
    CHECK(zeMemAllocDevice(ctx, &dd, bytes, 4096, dev, &p));
    return p;
}

static void *halloc(size_t bytes) {
    void *p = NULL;
    ze_host_mem_alloc_desc_t hd = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, NULL, 0};
    CHECK(zeMemAllocHost(ctx, &hd, bytes, 4096, &p));
    return p;
}

static void fill(void *p, uint32_t v, size_t n) { CHECK(zeCommandListAppendMemoryFill(imm, p, &v, 4, n, NULL, 0, NULL)); }
static void sync_imm(void) { CHECK(zeCommandListHostSynchronize(imm, UINT64_MAX)); }

static void copy(void *dst, const void *src, size_t n) {
    CHECK(zeCommandListAppendMemoryCopy(regular ? reg : imm, dst, src, n, NULL, 0, NULL));
}

static void flush(void) {
    if (!regular) return sync_imm();
    CHECK(zeCommandListClose(reg));
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &reg, NULL));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    CHECK(zeCommandListReset(reg));
}

// Words of [p, p + n) that differ from want(i) (i = word index from p); first mismatch reported.
static size_t verify(const char *what, const char *p, size_t n, uint32_t (*want)(size_t, const void *), const void *arg, uint32_t *host) {
    size_t bad = 0, first = SIZE_MAX;
    uint32_t got = 0, exp = 0;
    for (size_t done = 0; done < n; done += CHUNK) {
        size_t len = n - done < CHUNK ? n - done : CHUNK;
        CHECK(zeCommandListAppendMemoryCopy(imm, host, p + done, len, NULL, 0, NULL));
        sync_imm();
        for (size_t w = 0; w < len / 4; w++) {
            size_t i = done / 4 + w;
            if (host[w] != want(i, arg)) {
                if (first == SIZE_MAX) { first = i * 4; got = host[w]; exp = want(i, arg); }
                bad++;
            }
        }
    }
    if (bad) printf("    %s: %zu bad words, first at byte %zu holds 0x%08x, want 0x%08x\n", what, bad, first, got, exp);
    return bad;
}

static uint32_t want_poison(size_t i, const void *arg) { (void)i; (void)arg; return POISON; }
static uint32_t want_index(size_t i, const void *arg) { return (uint32_t)i + *(const uint32_t *)arg; }

static int straddle(size_t mib) {
    const size_t n = 84508672, head = 25044992; // layer 51's K copy at 165,056 tokens: 25 MB before 2^31
    size_t bytes = mib << 20;
    char *dst = dalloc(bytes), *src = dalloc(n);
    uint32_t *host = halloc(CHUNK);
    for (size_t done = 0; done < n; done += CHUNK) { // src word i = i
        size_t len = n - done < CHUNK ? n - done : CHUNK;
        for (size_t w = 0; w < len / 4; w++) host[w] = (uint32_t)(done / 4 + w);
        CHECK(zeCommandListAppendMemoryCopy(imm, src + done, host, len, NULL, 0, NULL));
        sync_imm();
    }
    int bad = 0;
    for (int b = 31; b <= 32; b++) {
        size_t edge = 1ull << b;
        if (edge + n > bytes) continue;
        for (regular = 0; regular < 2; regular++)
            for (int split = 0; split < 2; split++) {
                size_t start = edge - head;
                fill(dst, POISON, 4 << 20);
                fill(dst + start, POISON, n);
                sync_imm();
                if (split) {
                    copy(dst + start, src, head);
                    copy(dst + edge, src + head, n - head);
                } else copy(dst + start, src, n);
                flush();
                uint32_t zero = 0;
                size_t b0 = verify("offset 0..4 MiB", dst, 4 << 20, want_poison, NULL, host);
                size_t b1 = verify("copied range", dst + start, n, want_index, &zero, host);
                printf("  size %zu MiB va %p straddle 2^%d %s list %s: %zu bad words at 0..4 MiB, %zu in the copy\n", mib, (void *)dst, b,
                       regular ? "regular" : "immediate", split ? "split" : "whole", b0, b1);
                bad += (b0 + b1) != 0;
            }
    }
    CHECK(zeMemFree(ctx, host));
    CHECK(zeMemFree(ctx, src));
    CHECK(zeMemFree(ctx, dst));
    return bad;
}

// A copy whose destination crosses the buffer's first 4 GiB-aligned GPU address `head` bytes in (layer 27's K copy in
// the failing engine run crossed 0x...b00000000 at 8,019,968 bytes).
static int vacross(size_t mib) {
    const size_t n = 84508672, heads[] = {8019968, 2097152, 1 << 20, 40000000};
    size_t bytes = mib << 20;
    char *dst = dalloc(bytes), *src = dalloc(n);
    uint32_t *host = halloc(CHUNK);
    for (size_t done = 0; done < n; done += CHUNK) {
        size_t len = n - done < CHUNK ? n - done : CHUNK;
        for (size_t w = 0; w < len / 4; w++) host[w] = (uint32_t)(done / 4 + w);
        CHECK(zeCommandListAppendMemoryCopy(imm, src + done, host, len, NULL, 0, NULL));
        sync_imm();
    }
    uintptr_t base = (uintptr_t)dst, edge = (base + (1ull << 32) - 1) & ~((1ull << 32) - 1);
    int bad = 0;
    for (size_t h = 0; h < sizeof heads / sizeof heads[0]; h++) {
        size_t start = edge - base - heads[h];
        if (edge - base < heads[h] + (4 << 20) || start + n > bytes) { printf("  va %p: boundary at offset %zu leaves no room\n", (void *)dst, (size_t)(edge - base)); continue; }
        for (regular = 0; regular < 2; regular++) {
            fill(dst, POISON, 4 << 20);
            fill(dst + start, POISON, n);
            sync_imm();
            copy(dst + start, src, n);
            flush();
            uint32_t zero = 0;
            size_t b0 = verify("offset 0..4 MiB", dst, 4 << 20, want_poison, NULL, host);
            size_t b1 = verify("copied range", dst + start, n, want_index, &zero, host);
            printf("  size %zu MiB va %p boundary 0x%lx (offset %zu) crossed %zu bytes into the copy, %s list: %zu bad words at 0..4 MiB, %zu in the copy\n",
                   mib, (void *)dst, (unsigned long)edge, (size_t)(edge - base), heads[h], regular ? "regular" : "immediate", b0, b1);
            bad += (b0 + b1) != 0;
        }
    }
    CHECK(zeMemFree(ctx, host));
    CHECK(zeMemFree(ctx, src));
    CHECK(zeMemFree(ctx, dst));
    return bad;
}

// The save's regions: per layer, attention (i % 4 == 3) K rows, K scales, V rows, V scales; else GDN state, conv state.
struct region { size_t off, n; void *src; uint32_t id; };

static uint32_t want_const(size_t i, const void *arg) { (void)i; return *(const uint32_t *)arg; }

static int layout(size_t at) {
    const size_t ctxmax = 262144, kvr = 512, kvs = 8, rb = 24 * 128 * 128 * 4, cb = 3 * 5120 * 2;
    struct region regs[160];
    size_t nr = 0, o = 0;
    for (int i = 0; i < 64; i++) {
        if (i % 4 == 3) {
            for (int kv = 0; kv < 2; kv++) {
                char *cache = dalloc(ctxmax * (kvr + kvs)); // the slot's cache: rows, then the scales after MAX_CTX rows
                uint32_t id = (uint32_t)(nr + 1);
                fill(cache, id, at * kvr);
                fill(cache + ctxmax * kvr, id + 1, at * kvs);
                regs[nr++] = (struct region){o, at * kvr, cache, id};
                o += at * kvr;
                regs[nr++] = (struct region){o, at * kvs, cache + ctxmax * kvr, id + 1};
                o += at * kvs;
            }
        } else {
            char *rec = dalloc(rb), *conv = dalloc(cb);
            uint32_t id = (uint32_t)(nr + 1);
            fill(rec, id, rb);
            fill(conv, id + 1, cb);
            regs[nr++] = (struct region){o, rb, rec, id};
            regs[nr++] = (struct region){o + rb, cb, conv, id + 1};
            o += rb + cb;
        }
    }
    sync_imm();
    int bad = 0;
    for (regular = 0; regular < 2; regular++) {
        char *dst = dalloc(o);
        uint32_t *host = halloc(CHUNK);
        for (size_t k = 0; k < nr; k++) copy(dst + regs[k].off, regs[k].src, regs[k].n);
        flush();
        size_t b = 0;
        char what[64];
        for (size_t k = 0; k < nr; k++) {
            snprintf(what, sizeof what, "region %zu at byte %zu", k, regs[k].off);
            b += verify(what, dst + regs[k].off, regs[k].n, want_const, &regs[k].id, host);
        }
        printf("  layout at=%zu bytes=%zu va %p %s list: %zu bad words\n", at, o, (void *)dst, regular ? "regular" : "immediate", b);
        bad += b != 0;
        CHECK(zeMemFree(ctx, host));
        CHECK(zeMemFree(ctx, dst));
    }
    return bad;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: copy_alias straddle SIZE_MIB... | layout AT\n"); return 2; }
    CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    uint32_t n = 1;
    ze_driver_handle_t drv;
    CHECK(zeDriverGet(&n, &drv));
    uint32_t ndev = 0;
    CHECK(zeDeviceGet(drv, &ndev, NULL));
    n = 1;
    CHECK(zeDeviceGet(drv, &n, &dev));
    ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, NULL, 0};
    CHECK(zeContextCreate(drv, &cd, &ctx));
    ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL, 0, 0, ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
                                  ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    CHECK(zeCommandListCreateImmediate(ctx, dev, &qd, &imm));
    CHECK(zeCommandQueueCreate(ctx, dev, &qd, &queue));
    ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, NULL, 0, ZE_COMMAND_LIST_FLAG_IN_ORDER};
    CHECK(zeCommandListCreate(ctx, dev, &ld, &reg));
    printf("%u devices visible, testing the first\n", ndev);
    int bad = 0;
    if (!strcmp(argv[1], "straddle")) for (int a = 2; a < argc; a++) bad += straddle(strtoull(argv[a], NULL, 10));
    else if (!strcmp(argv[1], "layout")) bad += layout(strtoull(argv[2], NULL, 10));
    else if (!strcmp(argv[1], "vacross")) for (int a = 2; a < argc; a++) bad += vacross(strtoull(argv[a], NULL, 10));
    else return 2;
    printf("failing cases %d\n", bad);
    return bad != 0;
}
