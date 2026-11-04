// CUDA primitive helpers derived from NablaEngine 70d0c8c.
// Original collision translation: MuJoCo Warp / MuJoCo, Apache-2.0;
// Copyright 2025 The Newton Developers; Copyright DeepMind Technologies Limited.
// This file retains the original float arithmetic and adds no Warp kernels.
#define BOX_BOX_CCD 0
#define HAS_BOX 1

__device__ __forceinline__ void qmul(const float* a, const float* b, float* o) {
  float w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  float x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  float y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  float z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
  o[0] = w; o[1] = x; o[2] = y; o[3] = z;
}
__device__ __forceinline__ void qrot(const float* q, const float* v, float* o) {  // o = q v q^-1
  float tx = 2.f * (q[2] * v[2] - q[3] * v[1]), ty = 2.f * (q[3] * v[0] - q[1] * v[2]), tz = 2.f * (q[1] * v[1] - q[2] * v[0]);
  o[0] = v[0] + q[0] * tx + q[2] * tz - q[3] * ty;
  o[1] = v[1] + q[0] * ty + q[3] * tx - q[1] * tz;
  o[2] = v[2] + q[0] * tz + q[1] * ty - q[2] * tx;
}
__device__ __forceinline__ void qrot_inv(const float* q, const float* v, float* o) {
  float qc[4] = {q[0], -q[1], -q[2], -q[3]};
  qrot(qc, v, o);
}
__device__ __forceinline__ void cross(const float* a, const float* b, float* o) {
  o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0];
}
__device__ __forceinline__ float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Explicit roundings. The adjoint replays the forward's body poses, collision-geom poses and contact geometry (dist,
// pos, frame) and needs them bitwise (adjoint.py adj_poses / adj_contacts), in another kernel and, in the JAX library,
// another compilation unit. Written as plain C (qmul, qrot, ... above), the mul / add pairs are fused into FMAs by NVVM
// and ptxas depending on the surrounding code (the uses of a product, register pressure, -fmad, the architecture), so
// the same source can give results an ulp apart in two kernels. These intrinsics are never contracted (and __fmaf_rn
// is never split): an expression written with them has its IEEE value, the same in every kernel, compilation unit and
// -fmad setting (on one GPU: rsqrtf and sincosf are the architecture's; IEEE division and sqrtf, nvcc's default
// -prec-div / -prec-sqrt, are correctly rounded and need nothing). Every operation the replay reproduces is written
// with them: the *_rn helpers here and fwd_body (the poses, and with them the twists and accelerations), step.py's
// geom_pose, the contact geometry that collide_pair shares with the adjoint's adj_contact_geom (make_frame,
// sphere_sphere, plane_point, plane_capsule_frame, capsule_end, segment_point, capcap_*), the bounding tests, the limit
// test (limit_dists) and the rows' impedance (kbimp, imp_raw, contact_invw, efc_D). The forms are the ones ptxas chose
// for the plain C on sm_70, so the change left the V100 results bitwise as they were: a b + c d = fma(a, b, c d),
// a b - c d = fma(a, b, -(c d)), x +- a b = fma(+-a, b, x), sums from the left; a product with a second use is rounded
// first (W = Wp + round(qd aw) in fwd_body, p +- round(l a) for the two ends of a capsule). The rest of the engine
// (dynamics, solver, the adjoint's VJPs) keeps the plain C helpers above: it is not replayed. Test: the forward or the
// adjoint unit built with -fmad=false: the replay stays bitwise the forward's
#define RMUL(a, b) __fmul_rn((a), (b))
#define RADD(a, b) __fadd_rn((a), (b))
#define RSUB(a, b) __fsub_rn((a), (b))
#define RFMA(a, b, c) __fmaf_rn((a), (b), (c))   // a b + c, one rounding
#define RFMS(a, b, c) __fmaf_rn(-(a), (b), (c))  // c - a b, one rounding

__device__ __forceinline__ void qmul_rn(const float* a, const float* b, float* o) {
  float w = RFMS(a[3], b[3], RFMS(a[2], b[2], RFMA(a[0], b[0], -RMUL(a[1], b[1]))));
  float x = RFMS(a[3], b[2], RFMA(a[2], b[3], RFMA(a[0], b[1], RMUL(a[1], b[0]))));
  float y = RFMA(a[3], b[1], RFMA(a[2], b[0], RFMA(a[0], b[2], -RMUL(a[1], b[3]))));
  float z = RFMA(a[3], b[0], RFMS(a[2], b[1], RFMA(a[0], b[3], RMUL(a[1], b[2]))));
  o[0] = w; o[1] = x; o[2] = y; o[3] = z;
}
// o = q v q^-1: t = 2 qv x v, o = v + q0 t + qv x t
__device__ __forceinline__ void qrot_rn(const float* q, const float* v, float* o) {
  float tx = RMUL(2.f, RFMA(q[2], v[2], -RMUL(q[3], v[1])));
  float ty = RMUL(2.f, RFMA(q[3], v[0], -RMUL(q[1], v[2])));
  float tz = RMUL(2.f, RFMA(q[1], v[1], -RMUL(q[2], v[0])));
  o[0] = RFMS(q[3], ty, RFMA(q[2], tz, RFMA(q[0], tx, v[0])));
  o[1] = RFMS(q[1], tz, RFMA(q[3], tx, RFMA(q[0], ty, v[1])));
  o[2] = RFMS(q[2], tx, RFMA(q[1], ty, RFMA(q[0], tz, v[2])));
}
__device__ __forceinline__ void cross_rn(const float* a, const float* b, float* o) {
  o[0] = RFMA(a[1], b[2], -RMUL(a[2], b[1]));
  o[1] = RFMA(a[2], b[0], -RMUL(a[0], b[2]));
  o[2] = RFMA(a[0], b[1], -RMUL(a[1], b[0]));
}
__device__ __forceinline__ float dot3_rn(const float* a, const float* b) {
  return RFMA(a[2], b[2], RFMA(a[0], b[0], RMUL(a[1], b[1])));
}
__device__ __forceinline__ void zaxis_rn(const float* q, float* a) {  // a = q e_z: a plane's normal, a capsule's axis
  const float ez[3] = {0.f, 0.f, 1.f};
  qrot_rn(q, ez, a);
}
__device__ __forceinline__ void normalize3_rn(float* v) {
  float n = sqrtf(dot3_rn(v, v));
  if (n > 0.f) { v[0] /= n; v[1] /= n; v[2] /= n; }
}

// MuJoCo's orthogonals(a) / make_frame(a): rows n, b, c (explicit roundings, as all of the contact geometry below:
// the adjoint's replay computes the same bits, see source.DEVICE)
__device__ __forceinline__ void make_frame(const float* n, float* fr) {
  float b[3] = {0.f, 0.f, 0.f};
  if (-0.5f < n[1] && n[1] < 0.5f) b[1] = 1.f; else b[2] = 1.f;
  float d = dot3_rn(n, b);
  for (int k = 0; k < 3; k++) b[k] = RFMS(n[k], d, b[k]);
  normalize3_rn(b);
  float c[3];
  cross_rn(n, b, c);
  for (int k = 0; k < 3; k++) { fr[k] = n[k]; fr[3 + k] = b[k]; fr[6 + k] = c[k]; }
}
