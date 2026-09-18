// CUDA kernels for the numerical baseline.  The input signal and every stored
// node image are complex64 (float2), matching fast_bp/ffbp.py.  The suffix
// describes the arithmetic used inside a kernel:
//   fp64  : FP64 geometry, interpolation, phase and accumulation
//   fp32  : FP32 geometry, interpolation, phase and accumulation
//   mixed : FP64 geometry/phase, FP32 interpolation and accumulation

__device__ __forceinline__ void weights64(double t, double w[4]) {
    w[0] = -t * (t - 1.0) * (t - 2.0) / 6.0;
    w[1] = (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0;
    w[2] = -(t + 1.0) * t * (t - 2.0) / 2.0;
    w[3] = (t + 1.0) * t * (t - 1.0) / 6.0;
}

__device__ __forceinline__ void weights32(float t, float w[4]) {
    w[0] = -t * (t - 1.0f) * (t - 2.0f) / 6.0f;
    w[1] = (t + 1.0f) * (t - 1.0f) * (t - 2.0f) / 2.0f;
    w[2] = -(t + 1.0f) * t * (t - 2.0f) / 2.0f;
    w[3] = (t + 1.0f) * t * (t - 1.0f) / 6.0f;
}

__device__ __forceinline__ double2 sample1_fp64(
        const float2* row, int samples, double q) {
    const int i = (int)floor(q);
    if (i < 1 || i + 2 >= samples) return make_double2(0.0, 0.0);
    double w[4];
    weights64(q - (double)i, w);
    double2 z = make_double2(0.0, 0.0);
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const float2 v = row[i + a - 1];
        z.x += w[a] * (double)v.x;
        z.y += w[a] * (double)v.y;
    }
    return z;
}

__device__ __forceinline__ float2 sample1_fp32(
        const float2* row, int samples, float q) {
    const int i = (int)floorf(q);
    if (i < 1 || i + 2 >= samples) return make_float2(0.0f, 0.0f);
    float w[4];
    weights32(q - (float)i, w);
    float2 z = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const float2 v = row[i + a - 1];
        z.x += w[a] * v.x;
        z.y += w[a] * v.y;
    }
    return z;
}

__device__ __forceinline__ float2 sample1_mixed(
        const float2* row, int samples, double q) {
    const int i = (int)floor(q);
    if (i < 1 || i + 2 >= samples) return make_float2(0.0f, 0.0f);
    float w[4];
    weights32((float)(q - (double)i), w);
    float2 z = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const float2 v = row[i + a - 1];
        z.x += w[a] * v.x;
        z.y += w[a] * v.y;
    }
    return z;
}

__device__ __forceinline__ double2 sample2_fp64(
        const float2* image, int nr, int nu, double qr, double qu) {
    const int ir = (int)floor(qr);
    const int iu = (int)floor(qu);
    if (ir < 1 || iu < 1 || ir + 2 >= nr || iu + 2 >= nu)
        return make_double2(0.0, 0.0);
    double wr[4], wu[4];
    weights64(qr - (double)ir, wr);
    weights64(qu - (double)iu, wu);
    double2 z = make_double2(0.0, 0.0);
#pragma unroll
    for (int a = 0; a < 4; ++a) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const float2 v = image[(ir + a - 1) * nu + iu + b - 1];
            const double w = wr[a] * wu[b];
            z.x += w * (double)v.x;
            z.y += w * (double)v.y;
        }
    }
    return z;
}

__device__ __forceinline__ float2 sample2_fp32(
        const float2* image, int nr, int nu, float qr, float qu) {
    const int ir = (int)floorf(qr);
    const int iu = (int)floorf(qu);
    if (ir < 1 || iu < 1 || ir + 2 >= nr || iu + 2 >= nu)
        return make_float2(0.0f, 0.0f);
    float wr[4], wu[4];
    weights32(qr - (float)ir, wr);
    weights32(qu - (float)iu, wu);
    float2 z = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int a = 0; a < 4; ++a) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const float2 v = image[(ir + a - 1) * nu + iu + b - 1];
            const float w = wr[a] * wu[b];
            z.x += w * v.x;
            z.y += w * v.y;
        }
    }
    return z;
}

__device__ __forceinline__ float2 sample2_mixed(
        const float2* image, int nr, int nu, double qr, double qu) {
    const int ir = (int)floor(qr);
    const int iu = (int)floor(qu);
    if (ir < 1 || iu < 1 || ir + 2 >= nr || iu + 2 >= nu)
        return make_float2(0.0f, 0.0f);
    float wr[4], wu[4];
    weights32((float)(qr - (double)ir), wr);
    weights32((float)(qu - (double)iu), wu);
    float2 z = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int a = 0; a < 4; ++a) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const float2 v = image[(ir + a - 1) * nu + iu + b - 1];
            const float w = wr[a] * wu[b];
            z.x += w * v.x;
            z.y += w * v.y;
        }
    }
    return z;
}

extern "C" __global__ void bp_polar_fp64(
        const float2* data, int samples, const double* pos, int pulses,
        const double* origin, double r0, double dr, int nr,
        double u0, double du, int nu, double range_start, double data_dr,
        double k, const double* weight, const double* phase, int side,
        float2* output) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = nr * nu;
    if (p >= pixels) return;
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
    output[p] = make_float2((float)total.x, (float)total.y);
}

extern "C" __global__ void bp_polar_fp32(
        const float2* data, int samples, const float* pos, int pulses,
        const float* origin, float r0, float dr, int nr,
        float u0, float du, int nu, float range_start, float data_dr,
        float k, const float* weight, const float* phase, int side,
        float2* output) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = nr * nu;
    if (p >= pixels) return;
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
    output[p] = total;
}

extern "C" __global__ void bp_polar_mixed(
        const float2* data, int samples, const double* pos, int pulses,
        const double* origin, double r0, double dr, int nr,
        double u0, double du, int nu, double range_start, double data_dr,
        double k, const float* weight, const double* phase, int side,
        float2* output) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = nr * nu;
    if (p >= pixels) return;
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
    output[p] = total;
}

extern "C" __global__ void direct_chunk_fp64(
        const float2* data, int samples, const double* pos, int pulses,
        const double* origin, double r0, double dr, int nr,
        double u0, double du, int nu, double range_start, double data_dr,
        double k, const double* weight, const double* phase, int side,
        int pulse_begin, int pulse_end, double2* accumulator) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = nr * nu;
    if (p >= pixels) return;
    const int ir = p / nu;
    const int iu = p - ir * nu;
    const double r = r0 + (double)ir * dr;
    const double u = u0 + (double)iu * du;
    const double ground = sqrt(fmax(0.0, r * r - origin[2] * origin[2]));
    const double x = origin[0] + (double)side * ground * sqrt(fmax(0.0, 1.0 - u * u));
    const double y = origin[1] + ground * u;
    double2 total = accumulator[p];
    for (int n = pulse_begin; n < pulse_end && n < pulses; ++n) {
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
    accumulator[p] = total;
}

extern "C" __global__ void direct_chunk_fp32(
        const float2* data, int samples, const float* pos, int pulses,
        const float* origin, float r0, float dr, int nr,
        float u0, float du, int nu, float range_start, float data_dr,
        float k, const float* weight, const float* phase, int side,
        int pulse_begin, int pulse_end, float2* accumulator) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = nr * nu;
    if (p >= pixels) return;
    const int ir = p / nu;
    const int iu = p - ir * nu;
    const float r = r0 + (float)ir * dr;
    const float u = u0 + (float)iu * du;
    const float ground = sqrtf(fmaxf(0.0f, r * r - origin[2] * origin[2]));
    const float x = origin[0] + (float)side * ground * sqrtf(fmaxf(0.0f, 1.0f - u * u));
    const float y = origin[1] + ground * u;
    float2 total = accumulator[p];
    for (int n = pulse_begin; n < pulse_end && n < pulses; ++n) {
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
    accumulator[p] = total;
}

extern "C" __global__ void direct_chunk_mixed(
        const float2* data, int samples, const double* pos, int pulses,
        const double* origin, double r0, double dr, int nr,
        double u0, double du, int nu, double range_start, double data_dr,
        double k, const float* weight, const double* phase, int side,
        int pulse_begin, int pulse_end, float2* accumulator) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = nr * nu;
    if (p >= pixels) return;
    const int ir = p / nu;
    const int iu = p - ir * nu;
    const double r = r0 + (double)ir * dr;
    const double u = u0 + (double)iu * du;
    const double ground = sqrt(fmax(0.0, r * r - origin[2] * origin[2]));
    const double x = origin[0] + (double)side * ground * sqrt(fmax(0.0, 1.0 - u * u));
    const double y = origin[1] + ground * u;
    float2 total = accumulator[p];
    for (int n = pulse_begin; n < pulse_end && n < pulses; ++n) {
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
    accumulator[p] = total;
}

extern "C" __global__ void regrid_fp64(
        const float2* image, int src_nr, int src_nu,
        const double* src_origin, double src_r0, double src_dr,
        double src_u0, double src_du, const double* dst_origin,
        double r0, double dr, int nr, double u0, double du, int nu,
        double k, int side, float2* output) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nr * nu) return;
    const int ir = p / nu;
    const int iu = p - ir * nu;
    const double r = r0 + (double)ir * dr;
    const double u = u0 + (double)iu * du;
    const double gr = sqrt(fmax(0.0, r * r - dst_origin[2] * dst_origin[2]));
    const double x = dst_origin[0] + (double)side * gr * sqrt(fmax(0.0, 1.0 - u * u));
    const double y = dst_origin[1] + gr * u;
    const double dx = x - src_origin[0];
    const double dy = y - src_origin[1];
    const double g = sqrt(dx * dx + dy * dy);
    if (g == 0.0) { output[p] = make_float2(0.0f, 0.0f); return; }
    const double rr = sqrt(g * g + src_origin[2] * src_origin[2]);
    const double uu = dy / g;
    const double2 v = sample2_fp64(image, src_nr, src_nu,
                                   (rr - src_r0) / src_dr,
                                   (uu - src_u0) / src_du);
    double s, c;
    sincos(k * (rr - r), &s, &c);
    output[p] = make_float2((float)(v.x * c - v.y * s),
                            (float)(v.x * s + v.y * c));
}

extern "C" __global__ void regrid_fp32(
        const float2* image, int src_nr, int src_nu,
        const float* src_origin, float src_r0, float src_dr,
        float src_u0, float src_du, const float* dst_origin,
        float r0, float dr, int nr, float u0, float du, int nu,
        float k, int side, float2* output) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nr * nu) return;
    const int ir = p / nu;
    const int iu = p - ir * nu;
    const float r = r0 + (float)ir * dr;
    const float u = u0 + (float)iu * du;
    const float gr = sqrtf(fmaxf(0.0f, r * r - dst_origin[2] * dst_origin[2]));
    const float x = dst_origin[0] + (float)side * gr * sqrtf(fmaxf(0.0f, 1.0f - u * u));
    const float y = dst_origin[1] + gr * u;
    const float dx = x - src_origin[0];
    const float dy = y - src_origin[1];
    const float g = sqrtf(dx * dx + dy * dy);
    if (g == 0.0f) { output[p] = make_float2(0.0f, 0.0f); return; }
    const float rr = sqrtf(g * g + src_origin[2] * src_origin[2]);
    const float uu = dy / g;
    const float2 v = sample2_fp32(image, src_nr, src_nu,
                                  (rr - src_r0) / src_dr,
                                  (uu - src_u0) / src_du);
    float s, c;
    sincosf(k * (rr - r), &s, &c);
    output[p] = make_float2(v.x * c - v.y * s, v.x * s + v.y * c);
}

extern "C" __global__ void regrid_mixed(
        const float2* image, int src_nr, int src_nu,
        const double* src_origin, double src_r0, double src_dr,
        double src_u0, double src_du, const double* dst_origin,
        double r0, double dr, int nr, double u0, double du, int nu,
        double k, int side, float2* output) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nr * nu) return;
    const int ir = p / nu;
    const int iu = p - ir * nu;
    const double r = r0 + (double)ir * dr;
    const double u = u0 + (double)iu * du;
    const double gr = sqrt(fmax(0.0, r * r - dst_origin[2] * dst_origin[2]));
    const double x = dst_origin[0] + (double)side * gr * sqrt(fmax(0.0, 1.0 - u * u));
    const double y = dst_origin[1] + gr * u;
    const double dx = x - src_origin[0];
    const double dy = y - src_origin[1];
    const double g = sqrt(dx * dx + dy * dy);
    if (g == 0.0) { output[p] = make_float2(0.0f, 0.0f); return; }
    const double rr = sqrt(g * g + src_origin[2] * src_origin[2]);
    const double uu = dy / g;
    const float2 v = sample2_mixed(image, src_nr, src_nu,
                                   (rr - src_r0) / src_dr,
                                   (uu - src_u0) / src_du);
    double sd, cd;
    sincos(k * (rr - r), &sd, &cd);
    const float s = (float)sd;
    const float c = (float)cd;
    output[p] = make_float2(v.x * c - v.y * s, v.x * s + v.y * c);
}

extern "C" __global__ void add_float2_inplace(float2* left, const float2* right, int n) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const float2 a = left[p];
    const float2 b = right[p];
    left[p] = make_float2(a.x + b.x, a.y + b.y);
}

extern "C" __global__ void cast_double2_to_float2(const double2* input, float2* output, int n) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const double2 z = input[p];
    output[p] = make_float2((float)z.x, (float)z.y);
}

extern "C" __global__ void crop_float2(
        const float2* input, int src_nu, int r_begin, int u_begin,
        int nr, int nu, float2* output) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nr * nu) return;
    const int ir = p / nu;
    const int iu = p - ir * nu;
    output[p] = input[(ir + r_begin) * src_nu + iu + u_begin];
}
