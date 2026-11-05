// CUDA primitive helpers derived from NablaEngine 70d0c8c.
// Original collision translation: MuJoCo Warp / MuJoCo, Apache-2.0;
// Copyright 2025 The Newton Developers; Copyright DeepMind Technologies Limited.
// This file retains the original float arithmetic and adds no Warp kernels.
struct CapCap { float a1[3], a2[3], dif[3], ma, mb, mc, u, v, det; };
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

__device__ __forceinline__ float plane_point(const float* p1, const float* n, const float* c, float r, float* pos) {
  float dist = RSUB(RSUB(dot3_rn(c, n), dot3_rn(p1, n)), r), hp = RFMA(0.5f, dist, r);
  for (int k = 0; k < 3; k++) pos[k] = RFMS(n[k], hp, c[k]);
  return dist;
}

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

__device__ __forceinline__ void capsule_end(const float* p, const float* axis, float hl, int e, float* ce) {
  for (int k = 0; k < 3; k++) {
    float l = RMUL(axis[k], hl);
    ce[k] = e == 0 ? RADD(p[k], l) : RSUB(p[k], l);
  }
}

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

__device__ __forceinline__ void capcap_setup(const float* p1, const float* q1, float hl1, const float* p2,
                                             const float* q2, float hl2, CapCap& c) {
  zaxis_rn(q1, c.a1);
  zaxis_rn(q2, c.a2);
  for (int k = 0; k < 3; k++) { c.a1[k] = RMUL(c.a1[k], hl1); c.a2[k] = RMUL(c.a2[k], hl2); c.dif[k] = RSUB(p1[k], p2[k]); }
  c.ma = dot3_rn(c.a1, c.a1); c.mb = -dot3_rn(c.a1, c.a2); c.mc = dot3_rn(c.a2, c.a2);
  c.u = -dot3_rn(c.a1, c.dif); c.v = dot3_rn(c.a2, c.dif);
  c.det = RFMA(c.ma, c.mc, -RMUL(c.mb, c.mb));
}

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

__device__ __forceinline__ float capcap_core(const float* p1, const float* p2, const CapCap& c, float x1, float x2,
                                             float r1, float r2, float* dist, float* pos, float* n) {
  float v1[3], v2[3];
  for (int k = 0; k < 3; k++) { v1[k] = RFMA(c.a1[k], x1, p1[k]); v2[k] = RFMA(c.a2[k], x2, p2[k]); }
  return sphere_sphere(v1, r1, v2, r2, dist, pos, n);
}

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

__device__ __forceinline__ float safe_div(float x, float y) { return x / (y != 0.f ? y : 1e-15f); }

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
