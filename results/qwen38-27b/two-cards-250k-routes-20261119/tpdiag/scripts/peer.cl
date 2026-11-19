// Payload and flag are separate allocations, as in Forge's p2p_direct.cpp.
// The receiver polls only its own device memory and reads the payload in a
// fresh reduction kernel. No cross-device atomic operations are issued.
kernel void store4(global const uint *src, global uint *dst, uint words) {
    for (uint i = get_global_id(0); i < words; i += get_global_size(0)) {
        dst[i] = src[i];
    }
}

kernel void store16(global const uint4 *src, global uint4 *dst, uint groups) {
    for (uint i = get_global_id(0); i < groups; i += get_global_size(0)) {
        dst[i] = src[i];
    }
}

kernel void publish(global volatile uint *peer_flag, uint epoch) {
    *peer_flag = epoch;
}

kernel void await_epoch(global atomic_uint *flag, uint epoch, global uint *error) {
    for (uint attempt = 0; attempt < 100000; ++attempt) {
        if (atomic_load_explicit(flag, memory_order_acquire,
                                 memory_scope_all_svm_devices) == epoch) {
            *error = 0;
            return;
        }
    }
    *error = 1;
}

kernel void reduce_pair(global const float *src, global const float *peer,
                        global float *dst, uint words) {
    for (uint i = get_global_id(0); i < words; i += get_global_size(0)) {
        dst[i] = src[i] + peer[i];
    }
}

kernel void check_words(global const uint *data, uint words, uint expected,
                        global atomic_uint *errors) {
    uint mismatches = 0;
    for (uint i = get_global_id(0); i < words; i += get_global_size(0)) {
        mismatches += (data[i] != expected);
    }
    if (mismatches) {
        atomic_fetch_add_explicit(errors, mismatches, memory_order_relaxed,
                                  memory_scope_device);
    }
}

kernel void stamp(global uint *dst, uint value) {
    *dst = value;
}

// Chain test (added by the second tpdiag agent, 2026-11-19 00:40 CET).
// One "layer step" per rank: produce a partial, exchange it, reduce and check.
kernel void produce(global float4 *dst, uint quads, float value) {
    for (uint i = get_global_id(0); i < quads; i += get_global_size(0)) {
        dst[i] = (float4)(value);
    }
}

// TensorFold-style reduction: 16-byte loads of both operands (one may be on the peer card),
// sum, store locally, count mismatches against the expected value.
kernel void reduce_check(global const float4 *a, global const float4 *b, global float4 *dst,
                         uint quads, float expected, global atomic_uint *errors) {
    uint bad = 0;
    for (uint i = get_global_id(0); i < quads; i += get_global_size(0)) {
        float4 s = a[i] + b[i];
        dst[i] = s;
        bad += (s.x != expected) + (s.y != expected) + (s.z != expected) + (s.w != expected);
    }
    if (bad) {
        atomic_fetch_add_explicit(errors, bad, memory_order_relaxed, memory_scope_device);
    }
}

// Waits until the local flag reaches at least `epoch` (the peer may already be one step ahead).
kernel void await_ge(global atomic_uint *flag, uint epoch, global atomic_uint *timeouts) {
    for (uint attempt = 0; attempt < 4000000; ++attempt) {
        if (atomic_load_explicit(flag, memory_order_acquire,
                                 memory_scope_all_svm_devices) >= epoch) {
            return;
        }
    }
    atomic_fetch_add_explicit(timeouts, 1u, memory_order_relaxed, memory_scope_device);
}
