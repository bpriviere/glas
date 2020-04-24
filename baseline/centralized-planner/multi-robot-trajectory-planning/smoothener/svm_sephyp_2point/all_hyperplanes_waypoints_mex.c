/*
% inputs: paths: [DIM x NPTS x NROBOTS] array of points
%        (TODO: input times?)
%
% outputs: A: [DIM x NROBOTS x NROBOTS x (NPTS - 1)] array of 
%             hyperplane normal vectors for each robot-robot interaction
%             at each segment.
%          b: distance from origin for hyperplanes. i.e. a(:,...)^T x <= b.
%
function [A, b] = all_hyperplanes(paths, ellipsoid)
*/

#include "mex.h"
#include "solver.h"

#define SQR(x) ((x) * (x))

Vars vars;
Params params;
Workspace work;
Settings settings;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs != 2) {
		mexErrMsgTxt("need nrhs == 2\n");
	}
	if (nlhs != 2) {
		mexErrMsgTxt("need nlhs == 2\n");
	}
	mxArray const *pathsArr = prhs[0];
	mxArray const *ellipsoidArr = prhs[1];
	double const *paths = mxGetPr(prhs[0]);
	double const *ellipsoid = mxGetPr(prhs[1]);

	// paths: [DIM x NPTS x NROBOTS] array of points
	mwSize ndim = mxGetNumberOfDimensions(pathsArr);
	if (ndim != 3) {
		mexErrMsgTxt("paths input must be 3d array");
	}
	mwSize const *pathsDim = mxGetDimensions(pathsArr);
	if (pathsDim[0] != 3) {
		mexErrMsgTxt("first dimension of paths input must be 3");
	}
	mwSize const npts = pathsDim[1];
	mwSize const nrobots = pathsDim[2];

	// ellipsoid: 1x3 or 3x1
	if (mxGetNumberOfElements(ellipsoidArr) != 3) {
		mexErrMsgTxt("ellipsoid input must 3x1 or 1x3");
	}

	// A: [DIM x NROBOTS x NROBOTS x (NPTS - 1)] array of 
	// b: distance from origin for hyperplanes. i.e. a(:,...)^T x <= b.
	mwSize Adims[4] = {3, nrobots, nrobots, npts-1};
	mwSize bdims[3] = {nrobots, nrobots, npts-1};
	plhs[0] = mxCreateNumericArray(4, Adims, mxDOUBLE_CLASS, 0);
	plhs[1] = mxCreateNumericArray(3, bdims, mxDOUBLE_CLASS, 0);

	double *A = mxGetPr(plhs[0]);
	double *b = mxGetPr(plhs[1]);

	set_defaults();
	setup_indexing();


	params.Q[0] = SQR(ellipsoid[0]);
	params.Q[1] = SQR(ellipsoid[1]);
	params.Q[2] = SQR(ellipsoid[2]);
	params.Q[3] = 0.0;

	params.A[12] = 1.0;
	params.A[13] = 1.0;
	params.A[14] = -1.0;
	params.A[15] = -1.0;

	// paths: [DIM x NPTS x NROBOTS] array of points
	#define pathind(i1, i2, i3) paths[(i1) + 3*(i2) + npts*3*(i3)]
	// A: [DIM x NROBOTS x NROBOTS x (NPTS - 1)] array of 
	#define Aind(i1, i2, i3, i4) A[(i1) + 3*(i2) + 3*nrobots*(i3) + 3*nrobots*nrobots*(i4)]
	// b: distance from origin for hyperplanes. i.e. a(:,...)^T x <= b.
	#define bind(i1, i2, i3) b[(i1) + nrobots*(i2) + nrobots*nrobots*(i3)]

	for (int step = 0; step < (npts - 1); ++step) {
		for (int i = 0; i < nrobots; ++i) {
			Aind(0, i, i, step) = NAN;
			Aind(1, i, i, step) = NAN;
			Aind(2, i, i, step) = NAN;
			bind(i, i, step) = NAN;

			for (int j = i+1; j < nrobots; ++j) {
				params.A[0] = pathind(0, step, i);
				params.A[1] = pathind(0, step+1, i);
				params.A[2] = -pathind(0, step, j);
				params.A[3] = -pathind(0, step+1, j);

				params.A[4] = pathind(1, step, i);
				params.A[5] = pathind(1, step+1, i);
				params.A[6] = -pathind(1, step, j);
				params.A[7] = -pathind(1, step+1, j);

				params.A[8] = pathind(2, step, i);
				params.A[9] = pathind(2, step+1, i);
				params.A[10] = -pathind(2, step, j);
				params.A[11] = -pathind(2, step+1, j);

				set_defaults();
				settings.verbose = 0;
				setup_indexing();
				solve();

				double b0 = vars.beta[0];
				double b1 = vars.beta[1];
				double b2 = vars.beta[2];
				double b3 = vars.beta[3];

				double norm = sqrt(SQR(b0) + SQR(b1) + SQR(b2));

				Aind(0, i, j, step) = b0 / norm;
				Aind(1, i, j, step) = b1 / norm;
				Aind(2, i, j, step) = b2 / norm;
				bind(i, j, step) = -b3 / norm;

				Aind(0, j, i, step) = -b0 / norm;
				Aind(1, j, i, step) = -b1 / norm;
				Aind(2, j, i, step) = -b2 / norm;
				bind(j, i, step) = b3 / norm;
			}
		}
	}
}
