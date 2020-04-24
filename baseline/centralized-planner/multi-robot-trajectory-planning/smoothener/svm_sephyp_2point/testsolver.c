/* Produced by CVXGEN, 2016-11-30 17:09:32 -0500.  */
/* CVXGEN is Copyright (C) 2006-2012 Jacob Mattingley, jem@cvxgen.com. */
/* The code in this file is Copyright (C) 2006-2012 Jacob Mattingley. */
/* CVXGEN, or solvers produced by CVXGEN, cannot be used for commercial */
/* applications without prior written permission from Jacob Mattingley. */

/* Filename: testsolver.c. */
/* Description: Basic test harness for solver.c. */
#include "solver.h"
Vars vars;
Params params;
Workspace work;
Settings settings;
#define NUMTESTS 0
int main(int argc, char **argv) {
  int num_iters;
#if (NUMTESTS > 0)
  int i;
  double time;
  double time_per;
#endif
  set_defaults();
  setup_indexing();
  load_default_data();
  /* Solve problem instance for the record. */
  settings.verbose = 1;
  num_iters = solve();
#ifndef ZERO_LIBRARY_MODE
#if (NUMTESTS > 0)
  /* Now solve multiple problem instances for timing purposes. */
  settings.verbose = 0;
  tic();
  for (i = 0; i < NUMTESTS; i++) {
    solve();
  }
  time = tocq();
  printf("Timed %d solves over %.3f seconds.\n", NUMTESTS, time);
  time_per = time / NUMTESTS;
  if (time_per > 1) {
    printf("Actual time taken per solve: %.3g s.\n", time_per);
  } else if (time_per > 1e-3) {
    printf("Actual time taken per solve: %.3g ms.\n", 1e3*time_per);
  } else {
    printf("Actual time taken per solve: %.3g us.\n", 1e6*time_per);
  }
#endif
#endif
  return 0;
}
void load_default_data(void) {
  params.Q[0] = 1.5507979025745755;
  params.Q[1] = 1.7081478226181048;
  params.Q[2] = 1.2909047389129444;
  params.Q[3] = 1.510827605197663;
  params.A[0] = 1.5717878173906188;
  params.A[1] = 1.5851723557337523;
  params.A[2] = -1.497658758144655;
  params.A[3] = -1.171028487447253;
  params.A[4] = -1.7941311867966805;
  params.A[5] = -0.23676062539745413;
  params.A[6] = -1.8804951564857322;
  params.A[7] = -0.17266710242115568;
  params.A[8] = 0.596576190459043;
  params.A[9] = -0.8860508694080989;
  params.A[10] = 0.7050196079205251;
  params.A[11] = 0.3634512696654033;
  params.A[12] = -1.9040724704913385;
  params.A[13] = 0.23541635196352795;
  params.A[14] = -0.9629902123701384;
  params.A[15] = -0.3395952119597214;
}
