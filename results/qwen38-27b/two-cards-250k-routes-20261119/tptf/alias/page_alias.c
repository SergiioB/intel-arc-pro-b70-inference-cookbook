// Level Zero page-alias check: does any 2 MiB page of one large device allocation share memory with another?
// Usage: page_alias SIZE_MIB [SIZE_MIB ...]   (device = the first one Level Zero shows; pick it with ZE_AFFINITY_MASK)
// For each size: stamp page i's first 4 KiB with the value i (memory fill, in page order), read every page's first
// word back, and report pages whose word is not their own index. Then stamp in reverse order and check again.
#include <level_zero/ze_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { ze_result_t r_ = (x); if (r_) { fprintf(stderr, "%s:%d %s = 0x%x\n", __FILE__, __LINE__, #x, r_); exit(2); } } while (0)
#define PAGE (2ull << 20)
#define STAMP 4096

static int run(ze_context_handle_t ctx, ze_device_handle_t dev, ze_command_list_handle_t list, size_t mib, int reverse) {
    size_t bytes = mib << 20, pages = bytes / PAGE;
    void *buf = NULL;
    ze_device_mem_alloc_desc_t dd = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ze_relaxed_allocation_limits_exp_desc_t relaxed = {ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC, NULL,
                                                      ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE};
    if (bytes >= (4ull << 30)) dd.pNext = &relaxed;
    CHECK(zeMemAllocDevice(ctx, &dd, bytes, 4096, dev, &buf));
    uint32_t *host = malloc(pages * sizeof(uint32_t));
    for (size_t k = 0; k < pages; k++) {
        size_t i = reverse ? pages - 1 - k : k;
        uint32_t v = (uint32_t)i;
        CHECK(zeCommandListAppendMemoryFill(list, (char *)buf + i * PAGE, &v, sizeof v, STAMP, NULL, 0, NULL));
    }
    CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));
    for (size_t i = 0; i < pages; i++) CHECK(zeCommandListAppendMemoryCopy(list, &host[i], (char *)buf + i * PAGE, 4, NULL, 0, NULL));
    CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));
    int bad = 0;
    for (size_t i = 0; i < pages; i++) {
        if (host[i] == i) continue;
        if (bad < 8) printf("  page %zu (offset %zu MiB) holds %u (page %u's stamp)\n", i, i * 2, host[i], host[i]);
        bad++;
    }
    printf("size %zu MiB va %p pages %zu order %s: %d pages hold another page's stamp\n", mib, buf, pages,
           reverse ? "reverse" : "forward", bad);
    free(host);
    CHECK(zeMemFree(ctx, buf));
    return bad;
}

int main(int argc, char **argv) {
    CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    uint32_t n = 1;
    ze_driver_handle_t drv;
    CHECK(zeDriverGet(&n, &drv));
    n = 1;
    ze_device_handle_t dev;
    CHECK(zeDeviceGet(drv, &n, &dev));
    ze_device_properties_t props = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
    CHECK(zeDeviceGetProperties(dev, &props));
    printf("device %s maxMemAllocSize %llu\n", props.name, (unsigned long long)props.maxMemAllocSize);
    ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    ze_context_handle_t ctx;
    CHECK(zeContextCreate(drv, &cd, &ctx));
    ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL, 0, 0, ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
                                  ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t list;
    CHECK(zeCommandListCreateImmediate(ctx, dev, &qd, &list));
    int bad = 0;
    for (int a = 1; a < argc; a++)
        for (int reverse = 0; reverse < 2; reverse++) bad += run(ctx, dev, list, strtoull(argv[a], NULL, 10), reverse);
    CHECK(zeCommandListDestroy(list));
    CHECK(zeContextDestroy(ctx));
    printf("total aliased pages %d\n", bad);
    return bad != 0;
}
