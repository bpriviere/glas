/*
% inputs: paths: [DIM x PPVAL_PTS x NROBOTS] array of points
%           obs: [DIM x PPVAL_PTS x NOBS] array of points
%         ellipsoid
%
% outputs: A: [DIM x NROBOTS x NOBS] array of 
%             hyperplane normal vectors for each robot-robot interaction.
%          b: [NROBOTS x NOBS] array
%             distance from origin for hyperplanes. i.e. a(:,...)^T x <= b.
%
*/

#include "mex.h"
#include "solver.h"

#define SQR(x) ((x) * (x))
#define SET_SZ 32

Vars vars;
Params params;
Workspace work;
Settings settings;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs != 3) {
		mexErrMsgTxt("need nrhs == 3\n");
	}
	if (nlhs != 2) {
		mexErrMsgTxt("need nlhs == 2\n");
	}
	mxArray const *pathsArr = prhs[0];
	mxArray const *obsArr = prhs[1];
	mxArray const *ellipsoidArr = prhs[2];
	double const *paths = mxGetPr(pathsArr);
	double const *obs = mxGetPr(obsArr);
	double const *ellipsoid = mxGetPr(ellipsoidArr);

	// paths: [DIM x PPVAL_PTS x NROBOTS] array of points
	mwSize pathsNdim = mxGetNumberOfDimensions(pathsArr);
	if (pathsNdim != 3) {
		mexErrMsgTxt("paths input must be 3d array");
	}
	mwSize const *pathsDim = mxGetDimensions(pathsArr);
	if (pathsDim[0] != 3) {
		mexErrMsgTxt("first dimension of paths input must be 3");
	}
	if (pathsDim[1] != SET_SZ) {
		mexErrMsgTxt("second dimension of paths input must be SET_SZ");
	}
	mwSize const nRobots = pathsDim[2];

	mwSize obsNdim = mxGetNumberOfDimensions(obsArr);
	if (obsNdim != 3) {
		mexErrMsgTxt("obs input must be 3d array");
	}
	mwSize const *obsDim = mxGetDimensions(obsArr);
	if (obsDim[0] != 3) {
		mexErrMsgTxt("first dimension of paths input must be 3");
	}
	if (obsDim[1] != SET_SZ) {
		mexErrMsgTxt("second dimension of paths input must be SET_SZ");
	}
	mwSize const nObs = obsDim[2];

	// ellipsoid: 1x3 or 3x1
	if (mxGetNumberOfElements(ellipsoidArr) != 3) {
		mexErrMsgTxt("ellipsoid input must 3x1 or 1x3");
	}

	// A: [DIM x NROBOTS x NROBOTS] array of 
	// b: distance from origin for hyperplanes. i.e. a(:,...)^T x <= b.
	mwSize Adims[3] = {3, nRobots, nObs};
	mwSize bdims[2] = {nRobots, nObs};
	plhs[0] = mxCreateNumericArray(3, Adims, mxDOUBLE_CLASS, 0);
	plhs[1] = mxCreateNumericArray(2, bdims, mxDOUBLE_CLASS, 0);

	double *A = mxGetPr(plhs[0]);
	double *b = mxGetPr(plhs[1]);

	set_defaults();
	setup_indexing();

	params.Q[0] = SQR(ellipsoid[0]);
	params.Q[1] = SQR(ellipsoid[1]);
	params.Q[2] = SQR(ellipsoid[2]);
	params.Q[3] = 0.0;

	// paths: [DIM x PPVAL_PTS x NROBOTS] array of points
	#define pathind(i1, i2, i3) paths[(i1) + 3*(i2) + SET_SZ*3*(i3)]
	// obs: [DIM x PPVAL_PTS x NOBS] array of points
	#define obsind(i1, i2, i3) obs[(i1) + 3*(i2) + SET_SZ*3*(i3)]
	// A: [DIM x NROBOTS x NOBS] array of ... 
	#define Aind(i1, i2, i3) A[(i1) + 3*(i2) + 3*nRobots*(i3)]
	// b: distance from origin for hyperplanes. i.e. a(:,...)^T x <= b.
	#define bind(i1, i2) b[(i1) + nRobots*(i2)]

	#define paramsAind(i1, i2) params.A[(i1) + SET_SZ*2*(i2)]

	for (int i = 0; i < SET_SZ; ++i) {
		paramsAind(i, 3) = 1.0;
	}
	for (int i = SET_SZ; i < 2*SET_SZ; ++i) {
		paramsAind(i, 3) = -1.0;
	}

	for (int i = 0; i < nRobots; ++i) {
		for (int dim = 0; dim < 3; ++dim) {
			for (int p = 0; p < SET_SZ; ++p) {
				paramsAind(p, dim) = pathind(dim, p, i);
			}
		}

		for (int j = 0; j < nObs; ++j) {
			for (int dim = 0; dim < 3; ++dim) {
				for (int p = 0; p < SET_SZ; ++p) {
					paramsAind(SET_SZ + p, dim) = -obsind(dim, p, j);
				}
			}

			set_defaults();
			settings.verbose = 0;
			setup_indexing();
			solve();

			double b0 = vars.beta[0];
			double b1 = vars.beta[1];
			double b2 = vars.beta[2];
			double b3 = vars.beta[3];

			double invNorm = 1.0 / sqrt(SQR(b0) + SQR(b1) + SQR(b2));
			double a0 = invNorm * b0;
			double a1 = invNorm * b1;
			double a2 = invNorm * b2;
			double bb = -invNorm * b3;

			// make it a supporting hyperplane
			double minDist = 1000000000;
			for (int boxVtx = 0; boxVtx < 8; ++boxVtx) {
				double mul = a0 * obsind(0, boxVtx, j)
				           + a1 * obsind(1, boxVtx, j)
				           + a2 * obsind(2, boxVtx, j);
				minDist = fmin(minDist, mul - bb);
			}
			bb = bb + minDist;

			// A: [DIM x NROBOTS x NOBS] array of ... 
			Aind(0, i, j) = a0;
			Aind(1, i, j) = a1;
			Aind(2, i, j) = a2;
			bind(i, j) = bb;
		}
	}
}
