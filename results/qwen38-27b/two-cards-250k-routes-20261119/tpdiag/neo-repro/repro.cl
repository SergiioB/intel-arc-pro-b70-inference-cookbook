kernel void bump(global int *p, int v) { if (get_global_id(0) == 0) p[0] = p[0] + v; }
kernel void sum2(global const int *a, global const int *b, global int *out) { if (get_global_id(0) == 0) out[0] = a[0] + b[0]; }
