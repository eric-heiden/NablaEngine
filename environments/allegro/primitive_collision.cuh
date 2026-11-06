// Quaternion to cuboid axes. No other geometry types are supported.
__device__ __forceinline__ void quat2mat(const float* q, float* R) {
  float q00 = q[0] * q[0], q01 = q[0] * q[1], q02 = q[0] * q[2], q03 = q[0] * q[3], q11 = q[1] * q[1];
  float q12 = q[1] * q[2], q13 = q[1] * q[3], q22 = q[2] * q[2], q23 = q[2] * q[3], q33 = q[3] * q[3];
  R[0] = q00 + q11 - q22 - q33; R[1] = 2.f * (q12 - q03); R[2] = 2.f * (q13 + q02);
  R[3] = 2.f * (q12 + q03); R[4] = q00 - q11 + q22 - q33; R[5] = 2.f * (q23 - q01);
  R[6] = 2.f * (q13 - q02); R[7] = 2.f * (q23 + q01); R[8] = q00 - q11 - q22 + q33;
}
