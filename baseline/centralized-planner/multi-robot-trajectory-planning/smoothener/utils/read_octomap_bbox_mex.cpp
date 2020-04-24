/*
% input: string to octomap path
% output: bbox 3x2
*/

#include "mex.h"

#include <octomap/octomap.h>
#include <octomap/OcTree.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs != 1) {
		mexErrMsgTxt("need nrhs == 1\n");
	}
	if (nlhs != 1) {
		mexErrMsgTxt("need nlhs == 1\n");
	}

	/* input must be a string */
	if (!mxIsChar(prhs[0])) {
		mexErrMsgTxt("Input must be a string.\n");
	}

	char const *path = mxArrayToString(prhs[0]);

	if (path == NULL) {
		mexErrMsgTxt("Could not convert input to string.\n");
	}

	/* set C-style string output_buf to MATLAB mexFunction output*/
	plhs[0] = mxCreateDoubleMatrix(3, 2, mxREAL);
	double *b = mxGetPr(plhs[0]);

	octomap::OcTree input(path);
	input.getMetricMin(b[0], b[1], b[2]);
	input.getMetricMax(b[3], b[4], b[5]);
}
