// Parent fusion extension, compiled after the unchanged baseline source.
// Every helper is the baseline regrid arithmetic with an explicit pixel index.
// Each child result is rounded to float2 BEFORE the final __fadd_rn, preserving
// the storage rounding point and preventing cross-child multiply-add fusion.

__device__ __forceinline__ float2 resident_regrid_fp64(int p,
        const float2* image, int src_nr, int src_nu,
        const double* src_origin, double src_r0, double src_dr,
        double src_u0, double src_du, const double* dst_origin,
        double r0, double dr, int nr, double u0, double du, int nu,
        double k, int side) {
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
    if (g == 0.0) { return make_float2(0.0f, 0.0f); }
    const double rr = sqrt(g * g + src_origin[2] * src_origin[2]);
    const double uu = dy / g;
    const double2 v = sample2_fp64(image, src_nr, src_nu,
                                   (rr - src_r0) / src_dr,
                                   (uu - src_u0) / src_du);
    double s, c;
    sincos(k * (rr - r), &s, &c);
    return make_float2((float)(v.x * c - v.y * s),
                            (float)(v.x * s + v.y * c));
}


extern "C" __global__ void merge_regrid_fp64(
        const float2* left_image, int left_nr, int left_nu,
        const double* left_origin, double left_r0, double left_dr,
        double left_u0, double left_du,
        const float2* right_image, int right_nr, int right_nu,
        const double* right_origin, double right_r0, double right_dr,
        double right_u0, double right_du,
        const double* dst_origin, double r0, double dr, int nr,
        double u0, double du, int nu, double k, int side,
        float2* output) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nr * nu) return;
    const float2 left = resident_regrid_fp64(p,
        left_image, left_nr, left_nu, left_origin, left_r0, left_dr,
        left_u0, left_du, dst_origin, r0, dr, nr, u0, du, nu, k, side);
    const float2 right = resident_regrid_fp64(p,
        right_image, right_nr, right_nu, right_origin, right_r0, right_dr,
        right_u0, right_du, dst_origin, r0, dr, nr, u0, du, nu, k, side);
    output[p] = make_float2(__fadd_rn(left.x, right.x),
                            __fadd_rn(left.y, right.y));
}

__device__ __forceinline__ float2 resident_regrid_fp32(int p,
        const float2* image, int src_nr, int src_nu,
        const float* src_origin, float src_r0, float src_dr,
        float src_u0, float src_du, const float* dst_origin,
        float r0, float dr, int nr, float u0, float du, int nu,
        float k, int side) {
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
    if (g == 0.0f) { return make_float2(0.0f, 0.0f); }
    const float rr = sqrtf(g * g + src_origin[2] * src_origin[2]);
    const float uu = dy / g;
    const float2 v = sample2_fp32(image, src_nr, src_nu,
                                  (rr - src_r0) / src_dr,
                                  (uu - src_u0) / src_du);
    float s, c;
    sincosf(k * (rr - r), &s, &c);
    return make_float2(v.x * c - v.y * s, v.x * s + v.y * c);
}


extern "C" __global__ void merge_regrid_fp32(
        const float2* left_image, int left_nr, int left_nu,
        const float* left_origin, float left_r0, float left_dr,
        float left_u0, float left_du,
        const float2* right_image, int right_nr, int right_nu,
        const float* right_origin, float right_r0, float right_dr,
        float right_u0, float right_du,
        const float* dst_origin, float r0, float dr, int nr,
        float u0, float du, int nu, float k, int side,
        float2* output) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nr * nu) return;
    const float2 left = resident_regrid_fp32(p,
        left_image, left_nr, left_nu, left_origin, left_r0, left_dr,
        left_u0, left_du, dst_origin, r0, dr, nr, u0, du, nu, k, side);
    const float2 right = resident_regrid_fp32(p,
        right_image, right_nr, right_nu, right_origin, right_r0, right_dr,
        right_u0, right_du, dst_origin, r0, dr, nr, u0, du, nu, k, side);
    output[p] = make_float2(__fadd_rn(left.x, right.x),
                            __fadd_rn(left.y, right.y));
}

__device__ __forceinline__ float2 resident_regrid_mixed(int p,
        const float2* image, int src_nr, int src_nu,
        const double* src_origin, double src_r0, double src_dr,
        double src_u0, double src_du, const double* dst_origin,
        double r0, double dr, int nr, double u0, double du, int nu,
        double k, int side) {
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
    if (g == 0.0) { return make_float2(0.0f, 0.0f); }
    const double rr = sqrt(g * g + src_origin[2] * src_origin[2]);
    const double uu = dy / g;
    const float2 v = sample2_mixed(image, src_nr, src_nu,
                                   (rr - src_r0) / src_dr,
                                   (uu - src_u0) / src_du);
    double sd, cd;
    sincos(k * (rr - r), &sd, &cd);
    const float s = (float)sd;
    const float c = (float)cd;
    return make_float2(v.x * c - v.y * s, v.x * s + v.y * c);
}


extern "C" __global__ void merge_regrid_mixed(
        const float2* left_image, int left_nr, int left_nu,
        const double* left_origin, double left_r0, double left_dr,
        double left_u0, double left_du,
        const float2* right_image, int right_nr, int right_nu,
        const double* right_origin, double right_r0, double right_dr,
        double right_u0, double right_du,
        const double* dst_origin, double r0, double dr, int nr,
        double u0, double du, int nu, double k, int side,
        float2* output) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nr * nu) return;
    const float2 left = resident_regrid_mixed(p,
        left_image, left_nr, left_nu, left_origin, left_r0, left_dr,
        left_u0, left_du, dst_origin, r0, dr, nr, u0, du, nu, k, side);
    const float2 right = resident_regrid_mixed(p,
        right_image, right_nr, right_nu, right_origin, right_r0, right_dr,
        right_u0, right_du, dst_origin, r0, dr, nr, u0, du, nu, k, side);
    output[p] = make_float2(__fadd_rn(left.x, right.x),
                            __fadd_rn(left.y, right.y));
}
