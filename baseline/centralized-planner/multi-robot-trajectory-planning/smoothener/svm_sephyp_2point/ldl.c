/* Produced by CVXGEN, 2016-11-30 17:09:32 -0500.  */
/* CVXGEN is Copyright (C) 2006-2012 Jacob Mattingley, jem@cvxgen.com. */
/* The code in this file is Copyright (C) 2006-2012 Jacob Mattingley. */
/* CVXGEN, or solvers produced by CVXGEN, cannot be used for commercial */
/* applications without prior written permission from Jacob Mattingley. */

/* Filename: ldl.c. */
/* Description: Basic test harness for solver.c. */
#include "solver.h"
/* Be sure to place ldl_solve first, so storage schemes are defined by it. */
void ldl_solve(double *target, double *var) {
  int i;
  /* Find var = (L*diag(work.d)*L') \ target, then unpermute. */
  /* Answer goes into var. */
  /* Forward substitution. */
  /* Include permutation as we retrieve from target. Use v so we can unpermute */
  /* later. */
  work.v[0] = target[4];
  work.v[1] = target[5];
  work.v[2] = target[6];
  work.v[3] = target[7];
  work.v[4] = target[0];
  work.v[5] = target[1];
  work.v[6] = target[2];
  work.v[7] = target[3];
  work.v[8] = target[8]-work.L[0]*work.v[0]-work.L[1]*work.v[4]-work.L[2]*work.v[5]-work.L[3]*work.v[6]-work.L[4]*work.v[7];
  work.v[9] = target[9]-work.L[5]*work.v[1]-work.L[6]*work.v[4]-work.L[7]*work.v[5]-work.L[8]*work.v[6]-work.L[9]*work.v[7]-work.L[10]*work.v[8];
  work.v[10] = target[10]-work.L[11]*work.v[2]-work.L[12]*work.v[4]-work.L[13]*work.v[5]-work.L[14]*work.v[6]-work.L[15]*work.v[7]-work.L[16]*work.v[8]-work.L[17]*work.v[9];
  work.v[11] = target[11]-work.L[18]*work.v[3]-work.L[19]*work.v[4]-work.L[20]*work.v[5]-work.L[21]*work.v[6]-work.L[22]*work.v[7]-work.L[23]*work.v[8]-work.L[24]*work.v[9]-work.L[25]*work.v[10];
  /* Diagonal scaling. Assume correctness of work.d_inv. */
  for (i = 0; i < 12; i++)
    work.v[i] *= work.d_inv[i];
  /* Back substitution */
  work.v[10] -= work.L[25]*work.v[11];
  work.v[9] -= work.L[17]*work.v[10]+work.L[24]*work.v[11];
  work.v[8] -= work.L[10]*work.v[9]+work.L[16]*work.v[10]+work.L[23]*work.v[11];
  work.v[7] -= work.L[4]*work.v[8]+work.L[9]*work.v[9]+work.L[15]*work.v[10]+work.L[22]*work.v[11];
  work.v[6] -= work.L[3]*work.v[8]+work.L[8]*work.v[9]+work.L[14]*work.v[10]+work.L[21]*work.v[11];
  work.v[5] -= work.L[2]*work.v[8]+work.L[7]*work.v[9]+work.L[13]*work.v[10]+work.L[20]*work.v[11];
  work.v[4] -= work.L[1]*work.v[8]+work.L[6]*work.v[9]+work.L[12]*work.v[10]+work.L[19]*work.v[11];
  work.v[3] -= work.L[18]*work.v[11];
  work.v[2] -= work.L[11]*work.v[10];
  work.v[1] -= work.L[5]*work.v[9];
  work.v[0] -= work.L[0]*work.v[8];
  /* Unpermute the result, from v to var. */
  var[0] = work.v[4];
  var[1] = work.v[5];
  var[2] = work.v[6];
  var[3] = work.v[7];
  var[4] = work.v[0];
  var[5] = work.v[1];
  var[6] = work.v[2];
  var[7] = work.v[3];
  var[8] = work.v[8];
  var[9] = work.v[9];
  var[10] = work.v[10];
  var[11] = work.v[11];
#ifndef ZERO_LIBRARY_MODE
  if (settings.debug) {
    printf("Squared norm for solution is %.8g.\n", check_residual(target, var));
  }
#endif
}
void ldl_factor(void) {
  work.d[0] = work.KKT[0];
  if (work.d[0] < 0)
    work.d[0] = settings.kkt_reg;
  else
    work.d[0] += settings.kkt_reg;
  work.d_inv[0] = 1/work.d[0];
  work.L[0] = work.KKT[1]*work.d_inv[0];
  work.v[1] = work.KKT[2];
  work.d[1] = work.v[1];
  if (work.d[1] < 0)
    work.d[1] = settings.kkt_reg;
  else
    work.d[1] += settings.kkt_reg;
  work.d_inv[1] = 1/work.d[1];
  work.L[5] = (work.KKT[3])*work.d_inv[1];
  work.v[2] = work.KKT[4];
  work.d[2] = work.v[2];
  if (work.d[2] < 0)
    work.d[2] = settings.kkt_reg;
  else
    work.d[2] += settings.kkt_reg;
  work.d_inv[2] = 1/work.d[2];
  work.L[11] = (work.KKT[5])*work.d_inv[2];
  work.v[3] = work.KKT[6];
  work.d[3] = work.v[3];
  if (work.d[3] < 0)
    work.d[3] = settings.kkt_reg;
  else
    work.d[3] += settings.kkt_reg;
  work.d_inv[3] = 1/work.d[3];
  work.L[18] = (work.KKT[7])*work.d_inv[3];
  work.v[4] = work.KKT[8];
  work.d[4] = work.v[4];
  if (work.d[4] < 0)
    work.d[4] = settings.kkt_reg;
  else
    work.d[4] += settings.kkt_reg;
  work.d_inv[4] = 1/work.d[4];
  work.L[1] = (work.KKT[9])*work.d_inv[4];
  work.L[6] = (work.KKT[10])*work.d_inv[4];
  work.L[12] = (work.KKT[11])*work.d_inv[4];
  work.L[19] = (work.KKT[12])*work.d_inv[4];
  work.v[5] = work.KKT[13];
  work.d[5] = work.v[5];
  if (work.d[5] < 0)
    work.d[5] = settings.kkt_reg;
  else
    work.d[5] += settings.kkt_reg;
  work.d_inv[5] = 1/work.d[5];
  work.L[2] = (work.KKT[14])*work.d_inv[5];
  work.L[7] = (work.KKT[15])*work.d_inv[5];
  work.L[13] = (work.KKT[16])*work.d_inv[5];
  work.L[20] = (work.KKT[17])*work.d_inv[5];
  work.v[6] = work.KKT[18];
  work.d[6] = work.v[6];
  if (work.d[6] < 0)
    work.d[6] = settings.kkt_reg;
  else
    work.d[6] += settings.kkt_reg;
  work.d_inv[6] = 1/work.d[6];
  work.L[3] = (work.KKT[19])*work.d_inv[6];
  work.L[8] = (work.KKT[20])*work.d_inv[6];
  work.L[14] = (work.KKT[21])*work.d_inv[6];
  work.L[21] = (work.KKT[22])*work.d_inv[6];
  work.v[7] = work.KKT[23];
  work.d[7] = work.v[7];
  if (work.d[7] < 0)
    work.d[7] = settings.kkt_reg;
  else
    work.d[7] += settings.kkt_reg;
  work.d_inv[7] = 1/work.d[7];
  work.L[4] = (work.KKT[24])*work.d_inv[7];
  work.L[9] = (work.KKT[25])*work.d_inv[7];
  work.L[15] = (work.KKT[26])*work.d_inv[7];
  work.L[22] = (work.KKT[27])*work.d_inv[7];
  work.v[0] = work.L[0]*work.d[0];
  work.v[4] = work.L[1]*work.d[4];
  work.v[5] = work.L[2]*work.d[5];
  work.v[6] = work.L[3]*work.d[6];
  work.v[7] = work.L[4]*work.d[7];
  work.v[8] = work.KKT[28]-work.L[0]*work.v[0]-work.L[1]*work.v[4]-work.L[2]*work.v[5]-work.L[3]*work.v[6]-work.L[4]*work.v[7];
  work.d[8] = work.v[8];
  if (work.d[8] > 0)
    work.d[8] = -settings.kkt_reg;
  else
    work.d[8] -= settings.kkt_reg;
  work.d_inv[8] = 1/work.d[8];
  work.L[10] = (-work.L[6]*work.v[4]-work.L[7]*work.v[5]-work.L[8]*work.v[6]-work.L[9]*work.v[7])*work.d_inv[8];
  work.L[16] = (-work.L[12]*work.v[4]-work.L[13]*work.v[5]-work.L[14]*work.v[6]-work.L[15]*work.v[7])*work.d_inv[8];
  work.L[23] = (-work.L[19]*work.v[4]-work.L[20]*work.v[5]-work.L[21]*work.v[6]-work.L[22]*work.v[7])*work.d_inv[8];
  work.v[1] = work.L[5]*work.d[1];
  work.v[4] = work.L[6]*work.d[4];
  work.v[5] = work.L[7]*work.d[5];
  work.v[6] = work.L[8]*work.d[6];
  work.v[7] = work.L[9]*work.d[7];
  work.v[8] = work.L[10]*work.d[8];
  work.v[9] = work.KKT[29]-work.L[5]*work.v[1]-work.L[6]*work.v[4]-work.L[7]*work.v[5]-work.L[8]*work.v[6]-work.L[9]*work.v[7]-work.L[10]*work.v[8];
  work.d[9] = work.v[9];
  if (work.d[9] > 0)
    work.d[9] = -settings.kkt_reg;
  else
    work.d[9] -= settings.kkt_reg;
  work.d_inv[9] = 1/work.d[9];
  work.L[17] = (-work.L[12]*work.v[4]-work.L[13]*work.v[5]-work.L[14]*work.v[6]-work.L[15]*work.v[7]-work.L[16]*work.v[8])*work.d_inv[9];
  work.L[24] = (-work.L[19]*work.v[4]-work.L[20]*work.v[5]-work.L[21]*work.v[6]-work.L[22]*work.v[7]-work.L[23]*work.v[8])*work.d_inv[9];
  work.v[2] = work.L[11]*work.d[2];
  work.v[4] = work.L[12]*work.d[4];
  work.v[5] = work.L[13]*work.d[5];
  work.v[6] = work.L[14]*work.d[6];
  work.v[7] = work.L[15]*work.d[7];
  work.v[8] = work.L[16]*work.d[8];
  work.v[9] = work.L[17]*work.d[9];
  work.v[10] = work.KKT[30]-work.L[11]*work.v[2]-work.L[12]*work.v[4]-work.L[13]*work.v[5]-work.L[14]*work.v[6]-work.L[15]*work.v[7]-work.L[16]*work.v[8]-work.L[17]*work.v[9];
  work.d[10] = work.v[10];
  if (work.d[10] > 0)
    work.d[10] = -settings.kkt_reg;
  else
    work.d[10] -= settings.kkt_reg;
  work.d_inv[10] = 1/work.d[10];
  work.L[25] = (-work.L[19]*work.v[4]-work.L[20]*work.v[5]-work.L[21]*work.v[6]-work.L[22]*work.v[7]-work.L[23]*work.v[8]-work.L[24]*work.v[9])*work.d_inv[10];
  work.v[3] = work.L[18]*work.d[3];
  work.v[4] = work.L[19]*work.d[4];
  work.v[5] = work.L[20]*work.d[5];
  work.v[6] = work.L[21]*work.d[6];
  work.v[7] = work.L[22]*work.d[7];
  work.v[8] = work.L[23]*work.d[8];
  work.v[9] = work.L[24]*work.d[9];
  work.v[10] = work.L[25]*work.d[10];
  work.v[11] = work.KKT[31]-work.L[18]*work.v[3]-work.L[19]*work.v[4]-work.L[20]*work.v[5]-work.L[21]*work.v[6]-work.L[22]*work.v[7]-work.L[23]*work.v[8]-work.L[24]*work.v[9]-work.L[25]*work.v[10];
  work.d[11] = work.v[11];
  if (work.d[11] > 0)
    work.d[11] = -settings.kkt_reg;
  else
    work.d[11] -= settings.kkt_reg;
  work.d_inv[11] = 1/work.d[11];
#ifndef ZERO_LIBRARY_MODE
  if (settings.debug) {
    printf("Squared Frobenius for factorization is %.8g.\n", check_factorization());
  }
#endif
}
double check_factorization(void) {
  /* Returns the squared Frobenius norm of A - L*D*L'. */
  double temp, residual;
  /* Only check the lower triangle. */
  residual = 0;
  temp = work.KKT[8]-1*work.d[4]*1;
  residual += temp*temp;
  temp = work.KKT[13]-1*work.d[5]*1;
  residual += temp*temp;
  temp = work.KKT[18]-1*work.d[6]*1;
  residual += temp*temp;
  temp = work.KKT[23]-1*work.d[7]*1;
  residual += temp*temp;
  temp = work.KKT[0]-1*work.d[0]*1;
  residual += temp*temp;
  temp = work.KKT[2]-1*work.d[1]*1;
  residual += temp*temp;
  temp = work.KKT[4]-1*work.d[2]*1;
  residual += temp*temp;
  temp = work.KKT[6]-1*work.d[3]*1;
  residual += temp*temp;
  temp = work.KKT[1]-work.L[0]*work.d[0]*1;
  residual += temp*temp;
  temp = work.KKT[3]-work.L[5]*work.d[1]*1;
  residual += temp*temp;
  temp = work.KKT[5]-work.L[11]*work.d[2]*1;
  residual += temp*temp;
  temp = work.KKT[7]-work.L[18]*work.d[3]*1;
  residual += temp*temp;
  temp = work.KKT[28]-work.L[0]*work.d[0]*work.L[0]-1*work.d[8]*1-work.L[1]*work.d[4]*work.L[1]-work.L[2]*work.d[5]*work.L[2]-work.L[3]*work.d[6]*work.L[3]-work.L[4]*work.d[7]*work.L[4];
  residual += temp*temp;
  temp = work.KKT[29]-work.L[5]*work.d[1]*work.L[5]-1*work.d[9]*1-work.L[6]*work.d[4]*work.L[6]-work.L[7]*work.d[5]*work.L[7]-work.L[8]*work.d[6]*work.L[8]-work.L[9]*work.d[7]*work.L[9]-work.L[10]*work.d[8]*work.L[10];
  residual += temp*temp;
  temp = work.KKT[30]-work.L[11]*work.d[2]*work.L[11]-1*work.d[10]*1-work.L[12]*work.d[4]*work.L[12]-work.L[13]*work.d[5]*work.L[13]-work.L[14]*work.d[6]*work.L[14]-work.L[15]*work.d[7]*work.L[15]-work.L[16]*work.d[8]*work.L[16]-work.L[17]*work.d[9]*work.L[17];
  residual += temp*temp;
  temp = work.KKT[31]-work.L[18]*work.d[3]*work.L[18]-1*work.d[11]*1-work.L[19]*work.d[4]*work.L[19]-work.L[20]*work.d[5]*work.L[20]-work.L[21]*work.d[6]*work.L[21]-work.L[22]*work.d[7]*work.L[22]-work.L[23]*work.d[8]*work.L[23]-work.L[24]*work.d[9]*work.L[24]-work.L[25]*work.d[10]*work.L[25];
  residual += temp*temp;
  temp = work.KKT[9]-work.L[1]*work.d[4]*1;
  residual += temp*temp;
  temp = work.KKT[14]-work.L[2]*work.d[5]*1;
  residual += temp*temp;
  temp = work.KKT[19]-work.L[3]*work.d[6]*1;
  residual += temp*temp;
  temp = work.KKT[24]-work.L[4]*work.d[7]*1;
  residual += temp*temp;
  temp = work.KKT[10]-work.L[6]*work.d[4]*1;
  residual += temp*temp;
  temp = work.KKT[15]-work.L[7]*work.d[5]*1;
  residual += temp*temp;
  temp = work.KKT[20]-work.L[8]*work.d[6]*1;
  residual += temp*temp;
  temp = work.KKT[25]-work.L[9]*work.d[7]*1;
  residual += temp*temp;
  temp = work.KKT[11]-work.L[12]*work.d[4]*1;
  residual += temp*temp;
  temp = work.KKT[16]-work.L[13]*work.d[5]*1;
  residual += temp*temp;
  temp = work.KKT[21]-work.L[14]*work.d[6]*1;
  residual += temp*temp;
  temp = work.KKT[26]-work.L[15]*work.d[7]*1;
  residual += temp*temp;
  temp = work.KKT[12]-work.L[19]*work.d[4]*1;
  residual += temp*temp;
  temp = work.KKT[17]-work.L[20]*work.d[5]*1;
  residual += temp*temp;
  temp = work.KKT[22]-work.L[21]*work.d[6]*1;
  residual += temp*temp;
  temp = work.KKT[27]-work.L[22]*work.d[7]*1;
  residual += temp*temp;
  return residual;
}
void matrix_multiply(double *result, double *source) {
  /* Finds result = A*source. */
  result[0] = work.KKT[8]*source[0]+work.KKT[9]*source[8]+work.KKT[10]*source[9]+work.KKT[11]*source[10]+work.KKT[12]*source[11];
  result[1] = work.KKT[13]*source[1]+work.KKT[14]*source[8]+work.KKT[15]*source[9]+work.KKT[16]*source[10]+work.KKT[17]*source[11];
  result[2] = work.KKT[18]*source[2]+work.KKT[19]*source[8]+work.KKT[20]*source[9]+work.KKT[21]*source[10]+work.KKT[22]*source[11];
  result[3] = work.KKT[23]*source[3]+work.KKT[24]*source[8]+work.KKT[25]*source[9]+work.KKT[26]*source[10]+work.KKT[27]*source[11];
  result[4] = work.KKT[0]*source[4]+work.KKT[1]*source[8];
  result[5] = work.KKT[2]*source[5]+work.KKT[3]*source[9];
  result[6] = work.KKT[4]*source[6]+work.KKT[5]*source[10];
  result[7] = work.KKT[6]*source[7]+work.KKT[7]*source[11];
  result[8] = work.KKT[1]*source[4]+work.KKT[28]*source[8]+work.KKT[9]*source[0]+work.KKT[14]*source[1]+work.KKT[19]*source[2]+work.KKT[24]*source[3];
  result[9] = work.KKT[3]*source[5]+work.KKT[29]*source[9]+work.KKT[10]*source[0]+work.KKT[15]*source[1]+work.KKT[20]*source[2]+work.KKT[25]*source[3];
  result[10] = work.KKT[5]*source[6]+work.KKT[30]*source[10]+work.KKT[11]*source[0]+work.KKT[16]*source[1]+work.KKT[21]*source[2]+work.KKT[26]*source[3];
  result[11] = work.KKT[7]*source[7]+work.KKT[31]*source[11]+work.KKT[12]*source[0]+work.KKT[17]*source[1]+work.KKT[22]*source[2]+work.KKT[27]*source[3];
}
double check_residual(double *target, double *multiplicand) {
  /* Returns the squared 2-norm of lhs - A*rhs. */
  /* Reuses v to find the residual. */
  int i;
  double residual;
  residual = 0;
  matrix_multiply(work.v, multiplicand);
  for (i = 0; i < 4; i++) {
    residual += (target[i] - work.v[i])*(target[i] - work.v[i]);
  }
  return residual;
}
void fill_KKT(void) {
  work.KKT[8] = 2*params.Q[0];
  work.KKT[13] = 2*params.Q[1];
  work.KKT[18] = 2*params.Q[2];
  work.KKT[23] = 2*params.Q[3];
  work.KKT[0] = work.s_inv_z[0];
  work.KKT[2] = work.s_inv_z[1];
  work.KKT[4] = work.s_inv_z[2];
  work.KKT[6] = work.s_inv_z[3];
  work.KKT[1] = 1;
  work.KKT[3] = 1;
  work.KKT[5] = 1;
  work.KKT[7] = 1;
  work.KKT[28] = work.block_33[0];
  work.KKT[29] = work.block_33[0];
  work.KKT[30] = work.block_33[0];
  work.KKT[31] = work.block_33[0];
  work.KKT[9] = params.A[0];
  work.KKT[14] = params.A[4];
  work.KKT[19] = params.A[8];
  work.KKT[24] = params.A[12];
  work.KKT[10] = params.A[1];
  work.KKT[15] = params.A[5];
  work.KKT[20] = params.A[9];
  work.KKT[25] = params.A[13];
  work.KKT[11] = params.A[2];
  work.KKT[16] = params.A[6];
  work.KKT[21] = params.A[10];
  work.KKT[26] = params.A[14];
  work.KKT[12] = params.A[3];
  work.KKT[17] = params.A[7];
  work.KKT[22] = params.A[11];
  work.KKT[27] = params.A[15];
}
