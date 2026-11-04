// CUDA primitive helpers derived from NablaEngine 70d0c8c.
// Original collision translation: MuJoCo Warp / MuJoCo, Apache-2.0;
// Copyright 2025 The Newton Developers; Copyright DeepMind Technologies Limited.
// This file retains the original float arithmetic and adds no Warp kernels.
#define BOX_BOX_CCD 0
#define HAS_BOX 1
__device__ __forceinline__ float sphere_sphere(const float* p1, float r1, const float* p2, float r2, float* dist,
                                               float* pos, float* n) {
  float dir[3] = {RSUB(p2[0], p1[0]), RSUB(p2[1], p1[1]), RSUB(p2[2], p1[2])};
  float d = sqrtf(dot3_rn(dir, dir));
  if (d == 0.f) { n[0] = 1.f; n[1] = 0.f; n[2] = 0.f; } else { for (int k = 0; k < 3; k++) n[k] = dir[k] / d; }
  *dist = RSUB(d, RADD(r1, r2));
  float hp = RFMA(0.5f, *dist, r1);
  for (int k = 0; k < 3; k++) pos[k] = RFMA(n[k], hp, p1[k]);
  return d;
}

// The contact geometry of the mjlab model's primitives (MuJoCo Warp's collision_primitive_core), shared by the
// forward's collide_pair and the adjoint's replay (adjoint_contact.adj_contact_geom): the intermediates the VJP needs
// are outputs (the forward drops them).

// a sphere of radius r at c on the plane through p1 with normal n (plane-sphere; plane-capsule: an end of the
// capsule): the distance, and the contact point pos, midway in the penetration
__device__ __forceinline__ float plane_point(const float* p1, const float* n, const float* c, float r, float* pos) {
  float dist = RSUB(RSUB(dot3_rn(c, n), dot3_rn(p1, n)), r), hp = RFMA(0.5f, dist, r);
  for (int k = 0; k < 3; k++) pos[k] = RFMS(n[k], hp, c[k]);
  return dist;
}

// the frame of a plane-capsule contact: rows n, b = the capsule axis projected on the plane and normalized (make_frame's
// basis vector where the projection is shorter than 0.5), n x b; returns the projection's length
__device__ __forceinline__ float plane_capsule_frame(const float* n, const float* axis, float* fr) {
  float na = dot3_rn(n, axis), b[3];
  for (int k = 0; k < 3; k++) b[k] = RFMS(n[k], na, axis[k]);
  float bn = sqrtf(dot3_rn(b, b));
  if (bn > 0.f) for (int k = 0; k < 3; k++) b[k] /= bn;
  if (bn < 0.5f) {
    b[0] = 0.f; b[1] = 0.f; b[2] = 0.f;
    if (-0.5f < n[1] && n[1] < 0.5f) b[1] = 1.f; else b[2] = 1.f;
  }
  float c[3];
  cross_rn(n, b, c);
  for (int k = 0; k < 3; k++) { fr[k] = n[k]; fr[3 + k] = b[k]; fr[6 + k] = c[k]; }
  return bn;
}

// end e of a capsule (centre p, unit axis, half length hl): e = 0 the +axis end, 1 the -axis end
__device__ __forceinline__ void capsule_end(const float* p, const float* axis, float hl, int e, float* ce) {
  for (int k = 0; k < 3; k++) {
    float l = RMUL(axis[k], hl);
    ce[k] = e == 0 ? RADD(p[k], l) : RSUB(p[k], l);
  }
}

// sphere-capsule (MuJoCo Warp's sphere_capsule): the closest point pt = a + t ab of the capsule's segment (centre p2,
// unit axis, half length hl; a = p2 - hl axis, ab = 2 hl axis) to the sphere's centre p1, t = clamp(raw, 0, 1),
// raw = ap . ab / gam with ap = p1 - a and MuJoCo Warp's regularizer gam = ab . ab + 1e-6; returns raw
__device__ __forceinline__ float segment_point(const float* p1, const float* p2, const float* axis, float hl, float* a,
                                               float* ab, float* ap, float* gam, float* t, float* pt) {
  for (int k = 0; k < 3; k++) {
    a[k] = RFMS(axis[k], hl, p2[k]); ab[k] = RMUL(RMUL(2.f, axis[k]), hl); ap[k] = RSUB(p1[k], a[k]);
  }
  *gam = RADD(dot3_rn(ab, ab), 1e-6f);
  float raw = dot3_rn(ap, ab) / *gam;
  *t = fminf(fmaxf(raw, 0.f), 1.f);
  for (int k = 0; k < 3; k++) pt[k] = RFMA(*t, ab[k], a[k]);
  return raw;
}

// capsule-capsule (MuJoCo Warp's capsule_capsule): the segments p_k + x_k a_k, x_k in [-1, 1], with the scaled axes
// a_k = hl_k R_k e_z, dif = p1 - p2, the normal equations [ma mb; mb mc] x = (u, v) and det = ma mc - mb^2
struct CapCap { float a1[3], a2[3], dif[3], ma, mb, mc, u, v, det; };
__device__ __forceinline__ void capcap_setup(const float* p1, const float* q1, float hl1, const float* p2,
                                             const float* q2, float hl2, CapCap& c) {
  zaxis_rn(q1, c.a1);
  zaxis_rn(q2, c.a2);
  for (int k = 0; k < 3; k++) { c.a1[k] = RMUL(c.a1[k], hl1); c.a2[k] = RMUL(c.a2[k], hl2); c.dif[k] = RSUB(p1[k], p2[k]); }
  c.ma = dot3_rn(c.a1, c.a1); c.mb = -dot3_rn(c.a1, c.a2); c.mc = dot3_rn(c.a2, c.a2);
  c.u = -dot3_rn(c.a1, c.dif); c.v = dot3_rn(c.a2, c.dif);
  c.det = RFMA(c.ma, c.mc, -RMUL(c.mb, c.mb));
}
// the segment parameters where |det| >= 1e-15: the unconstrained minimum, then the clamp cascade. f1, f2: x1 / x2 free
// (not fixed at +-1; wp.clamp's gradient rule lo <= raw < hi for the clamp of x1 in the cascade's second stage), raw:
// x1 before that clamp (0 if it did not happen)
__device__ __forceinline__ void capcap_params(const CapCap& c, float* x1o, float* x2o, bool* f1, bool* f2, float* raw) {
  float x1 = RFMA(c.mc, c.u, -RMUL(c.mb, c.v)) / c.det, x2 = RFMA(c.ma, c.v, -RMUL(c.mb, c.u)) / c.det;
  *f1 = *f2 = true;
  *raw = 0.f;
  if (x1 > 1.f) { x1 = 1.f; x2 = RSUB(c.v, c.mb) / c.mc; *f1 = false; }
  else if (x1 < -1.f) { x1 = -1.f; x2 = RADD(c.v, c.mb) / c.mc; *f1 = false; }
  if (x2 > 1.f || x2 < -1.f) {
    bool hi = x2 > 1.f;
    x2 = hi ? 1.f : -1.f;
    *raw = hi ? RSUB(c.u, c.mb) / c.ma : RADD(c.u, c.mb) / c.ma;
    x1 = fminf(fmaxf(*raw, -1.f), 1.f);
    *f1 = *raw >= -1.f && *raw < 1.f;
    *f2 = false;
  }
  *x1o = x1; *x2o = x2;
}
// candidate e = 0..3 of the parallel case (|det| < 1e-15): an end of one segment (e = 0, 1: x1 = +1, -1; 2, 3: x2 =
// +1, -1) and the other's clamped closest parameter (raw before its clamp); f1, f2 as capcap_params
__device__ __forceinline__ void capcap_candidate(const CapCap& c, int e, float* x1, float* x2, bool* f1, bool* f2,
                                                 float* raw) {
  if (e < 2) {
    *x1 = e == 0 ? 1.f : -1.f;
    *raw = e == 0 ? RSUB(c.v, c.mb) / c.mc : RADD(c.v, c.mb) / c.mc;
    *x2 = fminf(fmaxf(*raw, -1.f), 1.f);
    *f1 = false;
    *f2 = *raw >= -1.f && *raw < 1.f;
  } else {
    *x2 = e == 2 ? 1.f : -1.f;
    *raw = e == 2 ? RSUB(c.u, c.mb) / c.ma : RADD(c.u, c.mb) / c.ma;
    *x1 = fminf(fmaxf(*raw, -1.f), 1.f);
    *f2 = false;
    *f1 = *raw >= -1.f && *raw < 1.f;
  }
}
// the sphere core on the closest points p_k + x_k a_k (radii r1, r2): returns nu, as sphere_sphere
__device__ __forceinline__ float capcap_core(const float* p1, const float* p2, const CapCap& c, float x1, float x2,
                                             float r1, float r2, float* dist, float* pos, float* n) {
  float v1[3], v2[3];
  for (int k = 0; k < 3; k++) { v1[k] = RFMA(c.a1[k], x1, p1[k]); v2[k] = RFMA(c.a2[k], x2, p2[k]); }
  return sphere_sphere(v1, r1, v2, r2, dist, pos, n);
}

// 3x3 helpers (row-major float[9]) for the box collisions, MuJoCo Warp's quat_to_mat convention
__device__ __forceinline__ void quat2mat(const float* q, float* R) {
  float q00 = q[0] * q[0], q01 = q[0] * q[1], q02 = q[0] * q[2], q03 = q[0] * q[3], q11 = q[1] * q[1];
  float q12 = q[1] * q[2], q13 = q[1] * q[3], q22 = q[2] * q[2], q23 = q[2] * q[3], q33 = q[3] * q[3];
  R[0] = q00 + q11 - q22 - q33; R[1] = 2.f * (q12 - q03); R[2] = 2.f * (q13 + q02);
  R[3] = 2.f * (q12 + q03); R[4] = q00 - q11 + q22 - q33; R[5] = 2.f * (q23 - q01);
  R[6] = 2.f * (q13 - q02); R[7] = 2.f * (q23 + q01); R[8] = q00 - q11 - q22 + q33;
}
__device__ __forceinline__ void mv3(const float* A, const float* v, float* o) {
  for (int i = 0; i < 3; i++) o[i] = A[3 * i] * v[0] + A[3 * i + 1] * v[1] + A[3 * i + 2] * v[2];
}
__device__ __forceinline__ void mtv3(const float* A, const float* v, float* o) {
  for (int i = 0; i < 3; i++) o[i] = A[i] * v[0] + A[3 + i] * v[1] + A[6 + i] * v[2];
}
__device__ __forceinline__ void mm3(const float* A, const float* B, float* C) {  // C = A B
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
}
__device__ __forceinline__ void mmt3(const float* A, const float* B, float* C) {  // C = A B'
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) C[3 * i + j] = A[3 * i] * B[3 * j] + A[3 * i + 1] * B[3 * j + 1] + A[3 * i + 2] * B[3 * j + 2];
}
__device__ __forceinline__ float safe_div(float x, float y) { return x / (y != 0.f ? y : 1e-15f); }

// sphere-box (MuJoCo Warp's sphere_box): distance, position, normal (from the sphere towards the box)
__device__ void sphere_box(const float* sp, float r, const float* bp, const float* R, const float* bs, float* dist,
                           float* pos, float* n) {
  float d[3] = {sp[0] - bp[0], sp[1] - bp[1], sp[2] - bp[2]}, c[3], cl[3], dir[3], pl[3], nl[3] = {0.f, 0.f, 0.f};
  mtv3(R, d, c);
  for (int k = 0; k < 3; k++) { cl[k] = fmaxf(-bs[k], fminf(bs[k], c[k])); dir[k] = cl[k] - c[k]; }
  float dd = sqrtf(dot3(dir, dir));
  if (dd <= 1e-15f) {
    float closest = 2.f * (bs[0] + bs[1] + bs[2]);
    int kk = 0;
    for (int i = 0; i < 6; i++) {
      float fd = fabsf(((i % 2) ? 1.f : -1.f) * bs[i / 2] - c[i / 2]);
      if (closest > fd) { closest = fd; kk = i; }
    }
    nl[kk / 2] = (kk % 2) ? -1.f : 1.f;
    for (int k = 0; k < 3; k++) pl[k] = c[k] + nl[k] * (r - closest) / 2.f;
    *dist = -closest - r;
  } else {
    for (int k = 0; k < 3; k++) { nl[k] = dir[k] / dd; pl[k] = 0.5f * (cl[k] + c[k] + nl[k] * r); }
    *dist = dd - r;
  }
  mv3(R, nl, n);
  float t[3];
  mv3(R, pl, t);
  for (int k = 0; k < 3; k++) pos[k] = bp[k] + t[k];
}

// capsule-box (MuJoCo Warp's capsule_box_witness + capsule_box): up to two sphere-box contacts along the segment
template <class S>
__device__ void capsule_box(S& s, int pair, const float* cp, const float* axisw, float cr, float hl, const float* bp,
                            const float* R, const float* bs) {
  float d[3] = {cp[0] - bp[0], cp[1] - bp[1], cp[2] - bp[2]}, pos[3], axis[3], ha[3];
  mtv3(R, d, pos);
  mtv3(R, axisw, axis);
  for (int k = 0; k < 3; k++) ha[k] = axis[k] * hl;
  int axisdir = int(ha[0] > 0.f) + 2 * int(ha[1] > 0.f) + 4 * int(ha[2] > 0.f);
  float bestdist = 1e32f, bestseg = -12.f, bestbox = 0.f;
  int cltype = -4, clface = -12, clcorner = -123, cledge = -123;
  for (int i = -1; i < 2; i += 2) {  // a capsule end closest to a face
    float tip[3], bpnt[3];
    int nout = 0, axout = -1;
    for (int j = 0; j < 3; j++) {
      tip[j] = pos[j] + float(i) * ha[j];
      bpnt[j] = tip[j];
      if (bpnt[j] < -bs[j]) { nout++; axout = j; bpnt[j] = -bs[j]; }
      else if (bpnt[j] > bs[j]) { nout++; axout = j; bpnt[j] = bs[j]; }
    }
    if (nout > 1) continue;
    float dist = 0.f;
    for (int j = 0; j < 3; j++) dist += (bpnt[j] - tip[j]) * (bpnt[j] - tip[j]);
    if (dist < bestdist) { bestdist = dist; bestseg = float(i); cltype = -2 + i; clface = axout; }
  }
  for (int i = 0; i < 8; i++)  // a box edge closest
    for (int j = 0; j < 3; j++) {
      if (i & (1 << j)) continue;
      float bpt[3] = {((i & 1) ? 1.f : -1.f) * bs[0], ((i & 2) ? 1.f : -1.f) * bs[1], ((i & 4) ? 1.f : -1.f) * bs[2]};
      bpt[j] = 0.f;
      float dif[3] = {bpt[0] - pos[0], bpt[1] - pos[1], bpt[2] - pos[2]};
      float u = -bs[j] * dif[j], v = dot3(ha, dif), ma = bs[j] * bs[j], mb = -bs[j] * ha[j], mc = hl * hl;
      float det = ma * mc - mb * mb;
      if (fabsf(det) < 1e-15f) continue;
      float idet = 1.f / det, x1 = (mc * u - mb * v) * idet, x2 = (ma * v - mb * u) * idet;
      int s1 = 1, s2 = 1;
      if (x1 > 1.f) { x1 = 1.f; s1 = 2; x2 = safe_div(v - mb, mc); }
      else if (x1 < -1.f) { x1 = -1.f; s1 = 0; x2 = safe_div(v + mb, mc); }
      bool over = x2 > 1.f;
      if (over || x2 < -1.f) {
        if (over) { x2 = 1.f; s2 = 2; x1 = safe_div(u - mb, ma); }
        else { x2 = -1.f; s2 = 0; x1 = safe_div(u + mb, ma); }
        if (x1 > 1.f) { x1 = 1.f; s1 = 2; } else if (x1 < -1.f) { x1 = -1.f; s1 = 0; }
      }
      for (int k = 0; k < 3; k++) dif[k] -= ha[k] * x2;
      dif[j] += bs[j] * x1;
      int ct = s1 * 3 + s2;
      float dsq = dot3(dif, dif);
      if (dsq < bestdist - 1e-15f) {
        bestdist = dsq; bestseg = x2; bestbox = x1;
        clcorner = i + (1 << j) * (ct / 6); cledge = j; cltype = ct;
      }
    }
  if (cltype == -4) return;
  float second = -4.f;
  if (cltype >= 0 && cltype / 3 != 1) {  // closest to a box corner
    int c1 = axisdir ^ clcorner;
    if (c1 != 0 && c1 != 7) {
      int mul, ax = 0, ax1 = 1, ax2 = 2;
      if (c1 == 1 || c1 == 2 || c1 == 4) mul = 1; else { mul = -1; c1 = 7 - c1; }
      if (c1 == 1) { ax = 0; ax1 = 1; ax2 = 2; } else if (c1 == 2) { ax = 1; ax1 = 2; ax2 = 0; }
      else if (c1 == 4) { ax = 2; ax1 = 0; ax2 = 1; }
      if (axis[ax] * axis[ax] > 0.5f) {
        float m = 2.f * safe_div(bs[ax], fabsf(ha[ax]));
        second = fminf(1.f - float(mul) * bestseg, m);
      } else {
        float m = 2.f * fminf(safe_div(bs[ax1], fabsf(ha[ax1])), safe_div(bs[ax2], fabsf(ha[ax2])));
        second = -fminf(1.f + float(mul) * bestseg, m);
      }
      second *= float(mul);
    }
  } else if (cltype >= 0 && cltype / 3 == 1) {  // on a box edge
    int c1 = (axisdir ^ clcorner) & (7 - (1 << cledge));
    if (c1 == 1 || c1 == 2 || c1 == 4) {
      int ax1 = 1, ax2 = 2, ax = cledge, mul;
      if (cledge == 0) { ax1 = 1; ax2 = 2; }
      if (cledge == 1) { ax1 = 2; ax2 = 0; }
      if (cledge == 2) { ax1 = 0; ax2 = 1; }
      if (fabsf(axis[ax1]) > fabsf(axis[ax2])) ax1 = ax2;
      ax2 = 3 - ax - ax1;
      if (c1 & (1 << ax2)) { mul = 1; second = 1.f - bestseg; } else { mul = -1; second = 1.f + bestseg; }
      float e1 = 2.f * safe_div(bs[ax2], fabsf(ha[ax2]));
      second = fminf(e1, second);
      float e2 = (((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0)) ? 1.f - bestbox : 1.f + bestbox;
      e1 = bs[ax] * safe_div(e2, fabsf(ha[ax]));
      second = fminf(e1, second);
      second *= float(mul);
    }
  } else if (clface != -1) {  // a capsule end closest to a face
    int mul = cltype == -3 ? 1 : -1;
    second = 2.f;
    float t1[3];
    for (int k = 0; k < 3; k++) t1[k] = pos[k] - ha[k] * float(mul);
    for (int i = 0; i < 3; i++) {
      if (i == clface) continue;
      float har = safe_div(float(mul), ha[i]);
      float e1 = (bs[i] - t1[i]) * har;
      if (0.f < e1 && e1 < second) second = e1;
      e1 = (-bs[i] - t1[i]) * har;
      if (0.f < e1 && e1 < second) second = e1;
    }
    second *= float(mul);
  }
  for (int e = 0; e < 2; e++) {
    if (e == 1 && !(second > -3.f)) break;
    float seg = e == 0 ? bestseg : bestseg + second, ce[3], dist, pc[3], n[3], fr[9];
    for (int k = 0; k < 3; k++) ce[k] = bp[k];
    float lc[3], t[3];
    for (int k = 0; k < 3; k++) lc[k] = pos[k] + ha[k] * seg;
    mv3(R, lc, t);
    for (int k = 0; k < 3; k++) ce[k] += t[k];
    sphere_box(ce, cr, bp, R, bs, &dist, pc, n);
    if (dist < 0.f) { make_frame(n, fr); add_contact(s, pair, e, dist, pc, fr); }
  }
}

__device__ __forceinline__ void rotmore_of(int f, float* r) {
  for (int k = 0; k < 9; k++) r[k] = 0.f;
  if (f == 0) { r[2] = -1.f; r[4] = 1.f; r[6] = 1.f; }
  else if (f == 1) { r[0] = 1.f; r[5] = -1.f; r[7] = 1.f; }
  else if (f == 2) { r[0] = 1.f; r[4] = 1.f; r[8] = 1.f; }
  else if (f == 3) { r[2] = 1.f; r[4] = 1.f; r[6] = -1.f; }
  else if (f == 4) { r[0] = 1.f; r[5] = 1.f; r[7] = -1.f; }
  else if (f == 5) { r[0] = -1.f; r[4] = 1.f; r[8] = -1.f; }
}

// box-box (MuJoCo Warp's box_box, margin 0): separating-axis test, then face-vertex or edge-edge clipping
#define BB_MAXPT 24
#ifndef BOX_BOX_CCD
#define BOX_BOX_CCD 1  // 1: MuJoCo Warp default (native CCD); 0: the separating-axis primitive (NATIVECCD disabled)
#endif
template <class S>
__device__ void box_box(S& s, int pair, const float* p1, const float* R1, const float* s1, const float* p2,
                        const float* R2, const float* s2) {
  const float margin = 0.f;
  float d[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]}, dn[3] = {-d[0], -d[1], -d[2]}, pos21[3], pos12[3];
  mtv3(R1, d, pos21);
  mtv3(R2, dn, pos12);
  float rot21[9], rot12[9], a21[9], a12[9], plen2[3], plen1[3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) rot21[3 * i + j] = R1[i] * R2[j] + R1[3 + i] * R2[3 + j] + R1[6 + i] * R2[6 + j];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { rot12[3 * i + j] = rot21[3 * j + i]; a21[3 * i + j] = fabsf(rot21[3 * i + j]); }
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) a12[3 * i + j] = a21[3 * j + i];
  mv3(a21, s2, plen2);
  mv3(a12, s1, plen1);
  float sep = margin + 3.f * (s1[0] + s2[0]) + 3.f * (s1[1] + s2[1]) + 3.f * (s1[2] + s2[2]);
  int code = -1;
  for (int i = 0; i < 3; i++) {
    float c1 = -fabsf(pos21[i]) + s1[i] + plen2[i], c2 = -fabsf(pos12[i]) + s2[i] + plen1[i];
    if (c1 < -margin || c2 < -margin) return;
    if (c1 < sep) { sep = c1; code = i + 3 * int(pos21[i] < 0.f); }
    if (c2 < sep) { sep = c2; code = i + 3 * int(pos12[i] < 0.f) + 6; }
  }
  float clnorm[3] = {0.f, 0.f, 0.f};
  bool inv = false;
  int cle1 = 0, cle2 = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      float ca[3];
      if (i == 0) { ca[0] = 0.f; ca[1] = -rot12[3 * j + 2]; ca[2] = rot12[3 * j + 1]; }
      else if (i == 1) { ca[0] = rot12[3 * j + 2]; ca[1] = 0.f; ca[2] = -rot12[3 * j]; }
      else { ca[0] = -rot12[3 * j + 1]; ca[1] = rot12[3 * j]; ca[2] = 0.f; }
      float cl = sqrtf(dot3(ca, ca));
      if (cl < 1e-15f) continue;
      for (int k = 0; k < 3; k++) ca[k] /= cl;
      float bd = dot3(pos21, ca), c3 = 0.f;
      for (int k = 0; k < 3; k++) {
        if (k != i) c3 += s1[k] * fabsf(ca[k]);
        if (k != j) c3 += s2[k] * a21[3 * i + 3 - k - j] / cl;
      }
      c3 -= fabsf(bd);
      if (c3 < -margin) return;
      if (c3 < sep * (1.f - 1e-12f)) {
        sep = c3; cle1 = 0; cle2 = 0;
        for (int k = 0; k < 3; k++) {
          if (k != i && (int(ca[k] > 0.f) ^ int(bd < 0.f))) cle1 += 1 << k;
          if (k != j && (int(rot21[3 * i + 3 - k - j] > 0.f) ^ int(bd < 0.f) ^ int((k - j + 3) % 3 == 1))) cle2 += 1 << k;
        }
        code = 12 + i * 3 + j;
        for (int k = 0; k < 3; k++) clnorm[k] = ca[k];
        inv = bd < 0.f;
      }
    }
  if (code == -1) return;
  float pts[BB_MAXPT][3], depth[BB_MAXPT], rw[9], pw[3], nrm[3], hz;
  int n = 0;
  if (code < 12) {  // face-vertex
    int face = code % 6, bi = code / 6;
    float rm[9], r[9], p[3], ss[3], sv[3], t[3];
    rotmore_of(face, rm);
    mm3(rm, bi ? rot12 : rot21, r);
    mv3(rm, bi ? pos12 : pos21, p);
    mv3(rm, bi ? s2 : s1, t);
    for (int k = 0; k < 3; k++) { ss[k] = fabsf(t[k]); sv[k] = bi ? s1[k] : s2[k]; }
    float lx = ss[0], ly = ss[1];
    hz = ss[2];
    p[2] -= hz;
    int clc = 0;
    for (int i = 0; i < 3; i++) if (r[6 + i] < 0.f) clc += 1 << i;
    float lp[3] = {p[0], p[1], p[2]};
    for (int i = 0; i < 3; i++)
      for (int k = 0; k < 3; k++) lp[k] += r[3 * k + i] * sv[i] * ((clc & (1 << i)) ? 1.f : -1.f);
    int dirs = 0;
    float cn1[3] = {0.f, 0.f, 0.f}, cn2[3] = {0.f, 0.f, 0.f};
    for (int i = 0; i < 3; i++)
      if (fabsf(r[6 + i]) < 0.5f) {
        float* cn = dirs ? cn2 : cn1;
        for (int k = 0; k < 3; k++) cn[k] = r[3 * k + i] * sv[i] * ((clc & (1 << i)) ? -2.f : 2.f);
        dirs++;
      }
    for (int i = 0; i < dirs * dirs; i++)
      for (int q = 0; q < 2; q++) {
        float la[3], lb[3];
        for (int k = 0; k < 3; k++) {
          la[k] = lp[k] + (i < 2 ? 0.f : (i == 2 ? cn1[k] : cn2[k]));
          lb[k] = (i == 0 || i == 3) ? cn1[k] : cn2[k];
        }
        if (fabsf(lb[q]) > 1e-15f) {
          float br = 1.f / lb[q];
          for (int j = -1; j < 2; j += 2) {
            float l = ss[q] * float(j), c1 = (l - la[q]) * br;
            if (c1 < 0.f || c1 > 1.f) continue;
            float c2 = la[1 - q] + lb[1 - q] * c1;
            if (fabsf(c2) > ss[1 - q]) continue;
            if (n < BB_MAXPT) { for (int k = 0; k < 3; k++) pts[n][k] = la[k] + c1 * lb[k]; }
            n++;
          }
        }
      }
    if (dirs == 2) {
      float ax = cn1[0], bx = cn2[0], ay = cn1[1], by = cn2[1], C = safe_div(1.f, ax * by - bx * ay);
      for (int i = 0; i < 4; i++) {
        float llx = (i / 2) ? lx : -lx, lly = (i % 2) ? ly : -ly, x = llx - lp[0], y = lly - lp[1];
        float u = (x * by - y * bx) * C, v = (y * ax - x * ay) * C;
        if (u > 0.f && v > 0.f && u < 1.f && v < 1.f) {
          if (n < BB_MAXPT) { pts[n][0] = llx; pts[n][1] = lly; pts[n][2] = lp[2] + u * cn1[2] + v * cn2[2]; }
          n++;
        }
      }
    }
    for (int i = 0; i < (1 << dirs); i++) {
      float tv[3];
      for (int k = 0; k < 3; k++) tv[k] = lp[k] + float(i & 1) * cn1[k] + float((i & 2) != 0) * cn2[k];
      if (tv[0] > -lx && tv[0] < lx && tv[1] > -ly && tv[1] < ly) {
        if (n < BB_MAXPT) { for (int k = 0; k < 3; k++) pts[n][k] = tv[k]; }
        n++;
      }
    }
    if (n > BB_MAXPT) atomicOr(&s.unsupported, 2);
    int m = min(n, BB_MAXPT);
    n = 0;
    for (int i = 0; i < m; i++) {
      if (pts[i][2] > margin) continue;
      for (int k = 0; k < 3; k++) pts[n][k] = pts[i][k];
      depth[n] = pts[n][2];
      pts[n][2] *= 0.5f;
      n++;
    }
    mmt3(bi ? R2 : R1, rm, rw);
    for (int k = 0; k < 3; k++) { pw[k] = bi ? p2[k] : p1[k]; nrm[k] = (bi ? -1.f : 1.f) * rw[3 * k + 2]; }
  } else {  // edge-edge
    int e1 = (code - 12) / 3, e2 = (code - 12) % 3;
    int ax1 = 1 - (e2 & 1), ax2 = 2 - (e2 & 2), pax1 = 1 - (e1 & 1), pax2 = 2 - (e1 & 2);
    if (a21[3 * e1 + ax1] < a21[3 * e1 + ax2]) { int t = ax1; ax1 = ax2; ax2 = t; }
    if (a12[3 * e2 + pax1] < a12[3 * e2 + pax2]) { int t = pax1; pax1 = pax2; pax2 = t; }
    float rm[9], p[3], rn[3], r[9], sv[3], t[3];
    rotmore_of((cle1 & (1 << pax2)) ? pax2 : pax2 + 3, rm);
    mv3(rm, pos21, p);
    mv3(rm, clnorm, rn);
    mm3(rm, rot21, r);
    mtv3(rm, s1, t);
    for (int k = 0; k < 3; k++) sv[k] = fabsf(t[k]);
    float lx = sv[0], ly = sv[1];
    hz = sv[2];
    p[2] -= hz;
    float pu[4][3];
    float g1 = (cle2 & (1 << ax1)) ? 1.f : -1.f, g2 = (cle2 & (1 << ax2)) ? 1.f : -1.f;
    for (int k = 0; k < 3; k++) {
      float b0 = p[k] + r[3 * k + ax1] * s2[ax1] * g1 + r[3 * k + ax2] * s2[ax2] * g2;
      float b2 = p[k] - r[3 * k + ax1] * s2[ax1] * g1 + r[3 * k + ax2] * s2[ax2] * g2;
      float de = r[3 * k + e2] * s2[e2];
      pu[0][k] = b0 + de; pu[1][k] = b0 - de; pu[2][k] = b2 + de; pu[3][k] = b2 - de;
    }
    float alp[3], acn1[3], acn2[3];
    for (int k = 0; k < 3; k++) { alp[k] = pu[0][k]; acn1[k] = pu[1][k] - pu[0][k]; acn2[k] = pu[2][k] - pu[0][k]; }
    if (fabsf(rn[2]) < 1e-15f) return;
    float sg = inv ? -1.f : 1.f, innorm = sg / rn[2], pp[4][3];
    for (int i = 0; i < 4; i++) {
      float cs = pu[i][2] * sg * innorm;
      for (int k = 0; k < 3; k++) pp[i][k] = pu[i][k] - rn[k] * cs;
    }
    float plp[3], pcn1[3], pcn2[3];
    for (int k = 0; k < 3; k++) { plp[k] = pp[0][k]; pcn1[k] = pp[1][k] - pp[0][k]; pcn2[k] = pp[2][k] - pp[0][k]; }
    for (int i = 0; i < 4; i++)
      for (int q = 0; q < 2; q++) {
        float la = plp[q] + (i < 2 ? 0.f : (i == 2 ? pcn1[q] : pcn2[q]));
        float lb = (i == 0 || i == 3) ? pcn1[q] : pcn2[q];
        float lc = plp[1 - q] + (i < 2 ? 0.f : (i == 2 ? pcn1[1 - q] : pcn2[1 - q]));
        float ld = (i == 0 || i == 3) ? pcn1[1 - q] : pcn2[1 - q];
        float lua[3], lub[3];
        for (int k = 0; k < 3; k++) {
          lua[k] = alp[k] + (i < 2 ? 0.f : (i == 2 ? acn1[k] : acn2[k]));
          lub[k] = (i == 0 || i == 3) ? acn1[k] : acn2[k];
        }
        if (fabsf(lb) > 1e-15f) {
          float br = 1.f / lb;
          for (int j = -1; j < 2; j += 2) {
            if (n == 8) break;
            float l = sv[q] * float(j), c1 = (l - la) * br;
            if (c1 < 0.f || c1 > 1.f) continue;
            float c2 = lc + ld * c1;
            if (fabsf(c2) > sv[1 - q]) continue;
            if ((lua[2] + lub[2] * c1) * innorm > margin) continue;
            for (int k = 0; k < 3; k++) pts[n][k] = lua[k] * 0.5f + c1 * lub[k] * 0.5f;
            pts[n][q] += 0.5f * l;
            pts[n][1 - q] += 0.5f * c2;
            depth[n] = pts[n][2] * innorm * 2.f;
            n++;
          }
        }
      }
    int nl = n;
    float ax = pcn1[0], bx = pcn2[0], ay = pcn1[1], by = pcn2[1], C = safe_div(1.f, ax * by - bx * ay);
    for (int i = 0; i < 4; i++) {
      if (n == 8) break;
      float llx = (i / 2) ? lx : -lx, lly = (i % 2) ? ly : -ly, x = llx - plp[0], y = lly - plp[1];
      float u = (x * by - y * bx) * C, v = (y * ax - x * ay) * C;
      if (nl == 0) { if ((u < 0.f || u > 1.f) && (v < 0.f || v > 1.f)) continue; }
      else if (u < 0.f || v < 0.f || u > 1.f || v > 1.f) continue;
      u = fminf(fmaxf(u, 0.f), 1.f);
      v = fminf(fmaxf(v, 0.f), 1.f);
      float w = 1.f - u - v, vt[3], q2[3] = {llx, lly, 0.f};
      for (int k = 0; k < 3; k++) vt[k] = pu[0][k] * w + pu[1][k] * u + pu[2][k] * v;
      float tc1 = 0.f;
      for (int k = 0; k < 3; k++) tc1 += (q2[k] - vt[k]) * (q2[k] - vt[k]);
      if (vt[2] > 0.f && tc1 > margin * margin) continue;
      for (int k = 0; k < 3; k++) pts[n][k] = 0.5f * (q2[k] + vt[k]);
      depth[n] = sqrtf(tc1) * (vt[2] < 0.f ? -1.f : 1.f);
      n++;
    }
    int nf = n;
    for (int i = 0; i < 4; i++) {
      if (n >= 8) break;
      float x = pu[i][0], y = pu[i][1];
      if (nl == 0 && nf != 0) { if ((x < -lx || x > lx) && (y < -ly || y > ly)) continue; }
      else if (x < -lx || x > lx || y < -ly || y > ly) continue;
      float c1 = 0.f;
      for (int j = 0; j < 2; j++) {
        if (pu[i][j] < -sv[j]) c1 += (pu[i][j] + sv[j]) * (pu[i][j] + sv[j]);
        else if (pu[i][j] > sv[j]) c1 += (pu[i][j] - sv[j]) * (pu[i][j] - sv[j]);
      }
      c1 += pu[i][2] * innorm * pu[i][2] * innorm;
      if (pu[i][2] > 0.f && c1 > margin * margin) continue;
      float tp[3] = {pu[i][0], pu[i][1], 0.f};
      for (int j = 0; j < 2; j++) {
        if (pu[i][j] < -sv[j]) tp[j] = -sv[j] * 0.5f;
        else if (pu[i][j] > sv[j]) tp[j] = sv[j] * 0.5f;
      }
      for (int k = 0; k < 3; k++) pts[n][k] = (tp[k] + pu[i][k]) * 0.5f;
      depth[n] = sqrtf(c1) * (pu[i][2] < 0.f ? -1.f : 1.f);
      n++;
    }
    mmt3(R1, rm, rw);
    float t2[3];
    mv3(rw, rn, t2);
    for (int k = 0; k < 3; k++) { pw[k] = p1[k]; nrm[k] = sg * t2[k]; }
  }
  if (n > 8) { atomicOr(&s.unsupported, 2); n = 8; }
  float fr[9];
  make_frame(nrm, fr);
#if BOX_BOX_CCD
  // MuJoCo Warp collides box-box with GJK/EPA (native CCD): one contact at the minimum-penetration axis, which is the
  // separating-axis result above, unless multiccd finds two faces within 0.0889 rad of facing each other or an edge
  // within that angle of lying on a face; those near-aligned cases are flagged (bit 8), not emulated
  const float ftol = 0.99605107f, etol = 0.08878297f;  // cos(0.0889), sin(0.0889)
  bool multi = false;
  if (code < 12) {  // face of one box against a vertex of the other: any axis of the vertex box near-perpendicular to n
    const float* Rv = (code / 6) ? R1 : R2;
    for (int k = 0; k < 3; k++) {
      float c = fabsf(Rv[k] * nrm[0] + Rv[3 + k] * nrm[1] + Rv[6 + k] * nrm[2]);
      if (c > ftol || c < etol) multi = true;
    }
    int best = -1;
    for (int i = 0; i < n; i++) if (depth[i] < 0.f && (best < 0 || depth[i] < depth[best])) best = i;
    if (multi) atomicOr(&s.unsupported, 8);
    if (best < 0) return;
    float pl[3] = {pts[best][0], pts[best][1], pts[best][2] + hz}, pc[3];
    mv3(rw, pl, pc);
    for (int k = 0; k < 3; k++) pc[k] += pw[k];
    add_contact(s, pair, 0, depth[best], pc, fr);
  } else {  // edge-edge: the midpoint of the two edges' closest points
    int e1 = (code - 12) / 3, e2 = (code - 12) % 3;
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        float c = fabsf(R1[a] * R2[b] + R1[3 + a] * R2[3 + b] + R1[6 + a] * R2[6 + b]);
        if (a != e1 && b != e2 && c > ftol) multi = true;      // adjacent faces facing each other
        if (a == e1 && b != e2 && c < etol) multi = true;      // edge 1 lying on a face of box 2
        if (b == e2 && a != e1 && c < etol) multi = true;      // edge 2 lying on a face of box 1
      }
    if (multi) atomicOr(&s.unsupported, 8);
    float l1[3], l2[3], c1[3], c2[3], d1[3], d2[3];
    for (int k = 0; k < 3; k++) {
      l1[k] = k == e1 ? 0.f : ((cle1 & (1 << k)) ? s1[k] : -s1[k]);
      l2[k] = k == e2 ? 0.f : ((cle2 & (1 << k)) ? s2[k] : -s2[k]);
    }
    mv3(R1, l1, c1);
    mv3(R2, l2, c2);
    for (int k = 0; k < 3; k++) {
      c1[k] += p1[k]; c2[k] += p2[k];
      d1[k] = R1[3 * k + e1] * s1[e1]; d2[k] = R2[3 * k + e2] * s2[e2];
    }
    // closest points of the segments c1 + t1 d1, c2 + t2 d2 with t in [-1, 1]
    float r[3] = {c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2]};
    float a11 = dot3(d1, d1), a12 = dot3(d1, d2), a22 = dot3(d2, d2), b1 = dot3(d1, r), b2 = dot3(d2, r);
    float det = a11 * a22 - a12 * a12, t1 = 0.f, t2 = 0.f;
    if (det > 1e-12f) t1 = fminf(fmaxf((a12 * b2 - a22 * b1) / det, -1.f), 1.f);
    t2 = fminf(fmaxf((b2 + a12 * t1) / a22, -1.f), 1.f);
    t1 = fminf(fmaxf((a12 * t2 - b1) / a11, -1.f), 1.f);
    float pc[3];
    for (int k = 0; k < 3; k++) pc[k] = 0.5f * (c1[k] + t1 * d1[k] + c2[k] + t2 * d2[k]);
    if (sep > 0.f) add_contact(s, pair, 0, -sep, pc, fr);
  }
#else
  for (int i = 0; i < n; i++) {
    if (!(depth[i] < 0.f)) continue;
    float pl[3] = {pts[i][0], pts[i][1], pts[i][2] + hz}, pc[3];
    mv3(rw, pl, pc);
    for (int k = 0; k < 3; k++) pc[k] += pw[k];
    add_contact(s, pair, i, depth[i], pc, fr);
  }
#endif
}
