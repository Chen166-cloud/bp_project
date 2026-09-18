// Direct-BP precision attribution probes. Append this file AFTER kernels.cu
// in the same translation unit: sample1_mixed is intentionally reused.
// These kernels have exactly the direct_chunk_mixed argument signature and
// use float2 accumulators. Compile with the baseline options, without
// --use_fast_math. They are diagnostic variants, not novel algorithms.
//
// direct_chunk_mixed_phase32:
//   Same FP64 geometry, sampling coordinates and phase expression as mixed;
//   cast the completed phase to FP32 and call the ordinary CUDA sincosf.
//   This isolates the FP64 sincos implementation plus phase-input precision.
//
// direct_chunk_mixed_phase32_wrapped:
//   First apply FP64 remainder(phi, 2*pi), then cast and call sincosf.
//   The additional range-reduction cost must be counted. This diagnostic
//   separates large-angle float quantization from sincosf evaluation error.
//   The stored 2*pi constant is finite precision, so this is not claimed to
//   reproduce libdevice FP64 argument reduction for arbitrarily large phi.

template <bool WrapPhase>
__device__ __forceinline__ void direct_chunk_phase32_probe_impl(
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
        double phi = k * (d - r) + phase[n];
        if (WrapPhase) {
            phi = remainder(phi, 6.283185307179586476925286766559005768);
        }
        float s, c;
        sincosf((float)phi, &s, &c);
        const float a = weight[n];
        total.x += a * (v.x * c - v.y * s);
        total.y += a * (v.x * s + v.y * c);
    }
    accumulator[p] = total;
}

extern "C" __global__ void direct_chunk_mixed_phase32(
        const float2* data, int samples, const double* pos, int pulses,
        const double* origin, double r0, double dr, int nr,
        double u0, double du, int nu, double range_start, double data_dr,
        double k, const float* weight, const double* phase, int side,
        int pulse_begin, int pulse_end, float2* accumulator) {
    direct_chunk_phase32_probe_impl<false>(
        data, samples, pos, pulses, origin, r0, dr, nr,
        u0, du, nu, range_start, data_dr, k, weight, phase, side,
        pulse_begin, pulse_end, accumulator);
}

extern "C" __global__ void direct_chunk_mixed_phase32_wrapped(
        const float2* data, int samples, const double* pos, int pulses,
        const double* origin, double r0, double dr, int nr,
        double u0, double du, int nu, double range_start, double data_dr,
        double k, const float* weight, const double* phase, int side,
        int pulse_begin, int pulse_end, float2* accumulator) {
    direct_chunk_phase32_probe_impl<true>(
        data, samples, pos, pulses, origin, r0, dr, nr,
        u0, du, nu, range_start, data_dr, k, weight, phase, side,
        pulse_begin, pulse_end, accumulator);
}

// MatX-inspired phase-LUT path, NOT a reproduction of the complete MatX BP
// operator. The idea is integer-bin reference phase plus fractional-bin phase:
// https://github.com/NVIDIA/MatX/blob/a108745201415478e17fa72e9e328dc0514ae7fb/include/matx/kernels/sar_bp.cuh
// This implementation retains this project's four-point Lagrange interpolation,
// FP64 geometry and FP64 range-bin coordinates. It uses ordinary sincosf,
// whereas the MatX LUT path uses the fast __sincosf intrinsic for the increment.
//
// Preparation metadata:
//   LUT shape/dtype: samples complex64 entries (samples * 8 bytes).
//   LUT dependency key: (samples, range_start, data_dr, k), all exact values.
//   Regenerate when any key component changes. Positions, pixel grid, pulse
//   weights, phase[n], and input data may change without invalidating the LUT.
//   Record LUT allocation and generation separately as one-time preparation;
//   additionally report preparation + execution when the dependency key changes.
//   The reference sincos(-k*r) is inside the chunk kernel, outside its pulse
//   loop, and remains included in every measured invocation.
//   phase[n] is added to the fractional-bin increment before the FP32 cast.
//   Large phase[n] can therefore still produce FP32 phase-input quantization.
//
// Algebra / sign check, for q=(d-range_start)/data_dr=i+f:
//   LUT[i] * exp(+i*(k*data_dr*f + phase[n])) * exp(-i*k*r)
// = exp(+i*(k*(d-r) + phase[n])).
// The identity is real-arithmetic exact; splitting the expression changes
// floating-point rounding, which must be assessed against the FP64 reference.

extern "C" __global__ void fill_range_phase_lut(
        int samples, double range_start, double data_dr, double k,
        float2* range_phase_lut) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= samples) return;
    const double bin_range = range_start + (double)i * data_dr;
    double s, c;
    sincos(k * bin_range, &s, &c);
    range_phase_lut[i] = make_float2((float)c, (float)s);
}

// Arguments are direct_chunk_mixed's arguments followed by range_phase_lut.
extern "C" __global__ void direct_chunk_mixed_phase_lut(
        const float2* data, int samples, const double* pos, int pulses,
        const double* origin, double r0, double dr, int nr,
        double u0, double du, int nu, double range_start, double data_dr,
        double k, const float* weight, const double* phase, int side,
        int pulse_begin, int pulse_end, float2* accumulator,
        const float2* range_phase_lut) {
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
    double ref_sd, ref_cd;
    sincos(-k * r, &ref_sd, &ref_cd);
    const float ref_s = (float)ref_sd;
    const float ref_c = (float)ref_cd;
    float2 total = accumulator[p];
    for (int n = pulse_begin; n < pulse_end && n < pulses; ++n) {
        const double dx = x - pos[3 * n];
        const double dy = y - pos[3 * n + 1];
        const double z = pos[3 * n + 2];
        const double d = sqrt(dx * dx + dy * dy + z * z);
        const double q = (d - range_start) / data_dr;
        const int i = (int)floor(q);
        // Exactly the valid-sample domain of sample1_mixed. Invalid samples
        // contribute zero, so skip both their data access and their LUT access.
        if (i < 1 || i + 2 >= samples) continue;
        const float2 v = sample1_mixed(data + (size_t)n * samples, samples, q);
        const double fraction = q - (double)i;
        float incr_s, incr_c;
        sincosf((float)(k * data_dr * fraction + phase[n]), &incr_s, &incr_c);
        const float2 base = range_phase_lut[i];
        const float bin_c = base.x * incr_c - base.y * incr_s;
        const float bin_s = base.x * incr_s + base.y * incr_c;
        const float c = bin_c * ref_c - bin_s * ref_s;
        const float s = bin_c * ref_s + bin_s * ref_c;
        const float a = weight[n];
        total.x += a * (v.x * c - v.y * s);
        total.y += a * (v.x * s + v.y * c);
    }
    accumulator[p] = total;
}
