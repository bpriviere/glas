% compiles all non-optional mex functions. "make" runs this for you.

cd svm_sephyp_2point
mex CFLAGS='$CFLAGS -std=c99' all_hyperplanes_waypoints_mex.c ldl.c matrix_support.c solver.c util.c
cd ..

cd svm_sephyp_32point
mex CFLAGS='$CFLAGS -std=c99' all_hyperplanes_pps_mex.c ldl.c matrix_support.c solver.c util.c
mex CFLAGS='$CFLAGS -std=c99' svm32_mex.c ldl.c matrix_support.c solver.c util.c
cd ..

cd utils
mex CXXFLAGS='$CXXFLAGS -std=c++11' read_octomap_bbox_mex.cpp /usr/lib/liboctomap.a /usr/lib/liboctomath.a
cd ..
