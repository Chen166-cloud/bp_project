// Compiled after kernels.cu and resident_kernels.cu.
// Node descriptor: [pulse_begin, pulse_end, nr, nu, side, left, right].
// Geometry descriptor: [r0, dr, u0, du]. A block never crosses node bounds.
// Leaf helpers preserve each arithmetic operation of the baseline kernels.

__device__ __forceinline__ float2 resident_leaf_fp64(int p,
        const float2* data, int samples, const double* pos, int pulses,
        const double* origin, double r0, double dr, int nr,
        double u0, double du, int nu, double range_start, double data_dr,
        double k, const double* weight, const double* phase, int side) {
    const int ir = p / nu;
    const int iu = p - ir * nu;
    const double r = r0 + (double)ir * dr;
    const double u = u0 + (double)iu * du;
    const double ground = sqrt(fmax(0.0, r * r - origin[2] * origin[2]));
    const double x = origin[0] + (double)side * ground * sqrt(fmax(0.0, 1.0 - u * u));
    const double y = origin[1] + ground * u;
    double2 total = make_double2(0.0, 0.0);
    for (int n = 0; n < pulses; ++n) {
        const double dx = x - pos[3 * n];
        const double dy = y - pos[3 * n + 1];
        const double z = pos[3 * n + 2];
        const double d = sqrt(dx * dx + dy * dy + z * z);
        const double2 v = sample1_fp64(data + (size_t)n * samples, samples,
                                       (d - range_start) / data_dr);
        double s, c;
        sincos(k * (d - r) + phase[n], &s, &c);
        const double a = weight[n];
        total.x += a * (v.x * c - v.y * s);
        total.y += a * (v.x * s + v.y * c);
    }
    return make_float2((float)total.x, (float)total.y);
}


extern "C" __global__ void batched_bp_polar_fp64(
        const float2* data, int samples, const double* pos,
        const double* weight, const double* phase,
        const double* origins, const int* nodes, const double* geometry,
        float2* const* images, const int* blocks,
        double range_start, double data_dr, double k) {
    const int node = blocks[2 * blockIdx.x];
    const int p = blocks[2 * blockIdx.x + 1] + threadIdx.x;
    const int* item = nodes + 7 * node;
    const int nr = item[2], nu = item[3];
    if (p >= nr * nu) return;
    const int begin = item[0];
    const double* geom = geometry + 4 * node;
    images[node][p] = resident_leaf_fp64(p,
        data + (size_t)begin * samples, samples, pos + 3 * begin,
        item[1] - begin, origins + 3 * node,
        geom[0], geom[1], nr, geom[2], geom[3], nu,
        range_start, data_dr, k, weight + begin, phase + begin, item[4]);
}

extern "C" __global__ void batched_merge_regrid_fp64(
        const double* origins, const int* nodes, const double* geometry,
        float2* const* images, const int* blocks, double k) {
    const int node = blocks[2 * blockIdx.x];
    const int p = blocks[2 * blockIdx.x + 1] + threadIdx.x;
    const int* item = nodes + 7 * node;
    const int nr = item[2], nu = item[3];
    if (p >= nr * nu) return;
    const int left = item[5], right = item[6];
    const int* ln = nodes + 7 * left;
    const int* rn = nodes + 7 * right;
    const double* geom = geometry + 4 * node;
    const double* lg = geometry + 4 * left;
    const double* rg = geometry + 4 * right;
    const float2 a = resident_regrid_fp64(p,
        images[left], ln[2], ln[3], origins + 3 * left,
        lg[0], lg[1], lg[2], lg[3], origins + 3 * node,
        geom[0], geom[1], nr, geom[2], geom[3], nu, k, item[4]);
    const float2 b = resident_regrid_fp64(p,
        images[right], rn[2], rn[3], origins + 3 * right,
        rg[0], rg[1], rg[2], rg[3], origins + 3 * node,
        geom[0], geom[1], nr, geom[2], geom[3], nu, k, item[4]);
    images[node][p] = make_float2(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y));
}

__device__ __forceinline__ float2 resident_leaf_fp32(int p,
        const float2* data, int samples, const float* pos, int pulses,
        const float* origin, float r0, float dr, int nr,
        float u0, float du, int nu, float range_start, float data_dr,
        float k, const float* weight, const float* phase, int side) {
    const int ir = p / nu;
    const int iu = p - ir * nu;
    const float r = r0 + (float)ir * dr;
    const float u = u0 + (float)iu * du;
    const float ground = sqrtf(fmaxf(0.0f, r * r - origin[2] * origin[2]));
    const float x = origin[0] + (float)side * ground * sqrtf(fmaxf(0.0f, 1.0f - u * u));
    const float y = origin[1] + ground * u;
    float2 total = make_float2(0.0f, 0.0f);
    for (int n = 0; n < pulses; ++n) {
        const float dx = x - pos[3 * n];
        const float dy = y - pos[3 * n + 1];
        const float z = pos[3 * n + 2];
        const float d = sqrtf(dx * dx + dy * dy + z * z);
        const float2 v = sample1_fp32(data + (size_t)n * samples, samples,
                                      (d - range_start) / data_dr);
        float s, c;
        sincosf(k * (d - r) + phase[n], &s, &c);
        const float a = weight[n];
        total.x += a * (v.x * c - v.y * s);
        total.y += a * (v.x * s + v.y * c);
    }
    return total;
}


extern "C" __global__ void batched_bp_polar_fp32(
        const float2* data, int samples, const float* pos,
        const float* weight, const float* phase,
        const float* origins, const int* nodes, const float* geometry,
        float2* const* images, const int* blocks,
        float range_start, float data_dr, float k) {
    const int node = blocks[2 * blockIdx.x];
    const int p = blocks[2 * blockIdx.x + 1] + threadIdx.x;
    const int* item = nodes + 7 * node;
    const int nr = item[2], nu = item[3];
    if (p >= nr * nu) return;
    const int begin = item[0];
    const float* geom = geometry + 4 * node;
    images[node][p] = resident_leaf_fp32(p,
        data + (size_t)begin * samples, samples, pos + 3 * begin,
        item[1] - begin, origins + 3 * node,
        geom[0], geom[1], nr, geom[2], geom[3], nu,
        range_start, data_dr, k, weight + begin, phase + begin, item[4]);
}

extern "C" __global__ void batched_merge_regrid_fp32(
        const float* origins, const int* nodes, const float* geometry,
        float2* const* images, const int* blocks, float k) {
    const int node = blocks[2 * blockIdx.x];
    const int p = blocks[2 * blockIdx.x + 1] + threadIdx.x;
    const int* item = nodes + 7 * node;
    const int nr = item[2], nu = item[3];
    if (p >= nr * nu) return;
    const int left = item[5], right = item[6];
    const int* ln = nodes + 7 * left;
    const int* rn = nodes + 7 * right;
    const float* geom = geometry + 4 * node;
    const float* lg = geometry + 4 * left;
    const float* rg = geometry + 4 * right;
    const float2 a = resident_regrid_fp32(p,
        images[left], ln[2], ln[3], origins + 3 * left,
        lg[0], lg[1], lg[2], lg[3], origins + 3 * node,
        geom[0], geom[1], nr, geom[2], geom[3], nu, k, item[4]);
    const float2 b = resident_regrid_fp32(p,
        images[right], rn[2], rn[3], origins + 3 * right,
        rg[0], rg[1], rg[2], rg[3], origins + 3 * node,
        geom[0], geom[1], nr, geom[2], geom[3], nu, k, item[4]);
    images[node][p] = make_float2(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y));
}

__device__ __forceinline__ float2 resident_leaf_mixed(int p,
        const float2* data, int samples, const double* pos, int pulses,
        const double* origin, double r0, double dr, int nr,
        double u0, double du, int nu, double range_start, double data_dr,
        double k, const float* weight, const double* phase, int side) {
    const int ir = p / nu;
    const int iu = p - ir * nu;
    const double r = r0 + (double)ir * dr;
    const double u = u0 + (double)iu * du;
    const double ground = sqrt(fmax(0.0, r * r - origin[2] * origin[2]));
    const double x = origin[0] + (double)side * ground * sqrt(fmax(0.0, 1.0 - u * u));
    const double y = origin[1] + ground * u;
    float2 total = make_float2(0.0f, 0.0f);
    for (int n = 0; n < pulses; ++n) {
        const double dx = x - pos[3 * n];
        const double dy = y - pos[3 * n + 1];
        const double z = pos[3 * n + 2];
        const double d = sqrt(dx * dx + dy * dy + z * z);
        const float2 v = sample1_mixed(data + (size_t)n * samples, samples,
                                       (d - range_start) / data_dr);
        double sd, cd;
        sincos(k * (d - r) + phase[n], &sd, &cd);
        const float s = (float)sd;
        const float c = (float)cd;
        const float a = weight[n];
        total.x += a * (v.x * c - v.y * s);
        total.y += a * (v.x * s + v.y * c);
    }
    return total;
}


extern "C" __global__ void batched_bp_polar_mixed(
        const float2* data, int samples, const double* pos,
        const float* weight, const double* phase,
        const double* origins, const int* nodes, const double* geometry,
        float2* const* images, const int* blocks,
        double range_start, double data_dr, double k) {
    const int node = blocks[2 * blockIdx.x];
    const int p = blocks[2 * blockIdx.x + 1] + threadIdx.x;
    const int* item = nodes + 7 * node;
    const int nr = item[2], nu = item[3];
    if (p >= nr * nu) return;
    const int begin = item[0];
    const double* geom = geometry + 4 * node;
    images[node][p] = resident_leaf_mixed(p,
        data + (size_t)begin * samples, samples, pos + 3 * begin,
        item[1] - begin, origins + 3 * node,
        geom[0], geom[1], nr, geom[2], geom[3], nu,
        range_start, data_dr, k, weight + begin, phase + begin, item[4]);
}

extern "C" __global__ void batched_merge_regrid_mixed(
        const double* origins, const int* nodes, const double* geometry,
        float2* const* images, const int* blocks, double k) {
    const int node = blocks[2 * blockIdx.x];
    const int p = blocks[2 * blockIdx.x + 1] + threadIdx.x;
    const int* item = nodes + 7 * node;
    const int nr = item[2], nu = item[3];
    if (p >= nr * nu) return;
    const int left = item[5], right = item[6];
    const int* ln = nodes + 7 * left;
    const int* rn = nodes + 7 * right;
    const double* geom = geometry + 4 * node;
    const double* lg = geometry + 4 * left;
    const double* rg = geometry + 4 * right;
    const float2 a = resident_regrid_mixed(p,
        images[left], ln[2], ln[3], origins + 3 * left,
        lg[0], lg[1], lg[2], lg[3], origins + 3 * node,
        geom[0], geom[1], nr, geom[2], geom[3], nu, k, item[4]);
    const float2 b = resident_regrid_mixed(p,
        images[right], rn[2], rn[3], origins + 3 * right,
        rg[0], rg[1], rg[2], rg[3], origins + 3 * node,
        geom[0], geom[1], nr, geom[2], geom[3], nu, k, item[4]);
    images[node][p] = make_float2(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y));
}
