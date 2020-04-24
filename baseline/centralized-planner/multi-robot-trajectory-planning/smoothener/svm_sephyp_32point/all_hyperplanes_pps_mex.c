/*
% inputs: paths: [DIM x PPVAL_PTS x NROBOTS] array of points
%         ellipsoid
%
% outputs: A: [DIM x NROBOTS x NROBOTS] array of 
%             hyperplane normal vectors for each robot-robot interaction.
%          b: [NROBOTS x NROBOTS] array
%             distance from origin for hyperplanes. i.e. a(:,...)^T x <= b.
%
function [A, b] = all_hyperplanes(paths, ellipsoid)
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

	// paths: [DIM x PPVAL_PTS x NROBOTS] array of points
	mwSize ndim = mxGetNumberOfDimensions(pathsArr);
	if (ndim != 3) {
		mexErrMsgTxt("paths input must be 3d array");
	}
	mwSize const *pathsDim = mxGetDimensions(pathsArr);
	if (pathsDim[0] != 3) {
		mexErrMsgTxt("first dimension of paths input must be 3");
	}
	if (pathsDim[1] != SET_SZ) {
		mexErrMsgTxt("second dimension of paths input must be SET_SZ");
	}
	mwSize const nrobots = pathsDim[2];

	// ellipsoid: 1x3 or 3x1
	if (mxGetNumberOfElements(ellipsoidArr) != 3) {
		mexErrMsgTxt("ellipsoid input must 3x1 or 1x3");
	}

	// A: [DIM x NROBOTS x NROBOTS] array of 
	// b: distance from origin for hyperplanes. i.e. a(:,...)^T x <= b.
	mwSize Adims[3] = {3, nrobots, nrobots};
	mwSize bdims[2] = {nrobots, nrobots};
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
	// A: [DIM x NROBOTS x NROBOTS] array of ... 
	#define Aind(i1, i2, i3) A[(i1) + 3*(i2) + 3*nrobots*(i3)]
	// b: distance from origin for hyperplanes. i.e. a(:,...)^T x <= b.
	#define bind(i1, i2) b[(i1) + nrobots*(i2)]

	#define paramsAind(i1, i2) params.A[(i1) + SET_SZ*2*(i2)]

	for (int i = 0; i < SET_SZ; ++i) {
		paramsAind(i, 3) = 1.0;
	}
	for (int i = SET_SZ; i < 2*SET_SZ; ++i) {
		paramsAind(i, 3) = -1.0;
	}

	for (int i = 0; i < nrobots; ++i) {
		Aind(0, i, i) = NAN;
		Aind(1, i, i) = NAN;
		Aind(2, i, i) = NAN;
		bind(i, i) = NAN;

		for (int dim = 0; dim < 3; ++dim) {
			for (int p = 0; p < SET_SZ; ++p) {
				paramsAind(p, dim) = pathind(dim, p, i);
			}
		}

		for (int j = i+1; j < nrobots; ++j) {
			for (int dim = 0; dim < 3; ++dim) {
				for (int p = 0; p < SET_SZ; ++p) {
					paramsAind(SET_SZ + p, dim) = -pathind(dim, p, j);
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

			double norm = sqrt(SQR(b0) + SQR(b1) + SQR(b2));

			// A: [DIM x NROBOTS x NROBOTS] array of ... 
			Aind(0, i, j) = b0 / norm;
			Aind(1, i, j) = b1 / norm;
			Aind(2, i, j) = b2 / norm;
			bind(i, j) = -b3 / norm;

			Aind(0, j, i) = -b0 / norm;
			Aind(1, j, i) = -b1 / norm;
			Aind(2, j, i) = -b2 / norm;
			bind(j, i) = b3 / norm;
		}
	}
}
