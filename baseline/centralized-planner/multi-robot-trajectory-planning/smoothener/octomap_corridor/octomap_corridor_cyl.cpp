#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include "octomap/OcTree.h"

#include "array2d.hpp"


// cvxgen requires that we define these globals :(
#include "cvxgen/solver.h"
Vars vars;
Params params;
Workspace work;
Settings settings;


// TODO: make it an input, maybe
// static float const OCCUPANCY_THRESHOLD = 0.95;

// TODO: these are fixed to avoid needing 4d array class, but should be variable
static int const DIM = 3;
static int const PP_DEGREE = 7;
static int const PP_SIZE = PP_DEGREE + 1;
static int const SAMPLES_PER_PIECE = 32;

// ax <= b
struct halfspace
{
	float a[3];
	float b;
};

using poly_piece = std::array<std::array<float, PP_SIZE>, DIM>;
using poly_sample = std::array<std::array<float, SAMPLES_PER_PIECE>, DIM>;
using polyhedron = std::vector<halfspace>;

template <typename T>
T sqr(T const &x)
{
	return x * x;
}

using p3d = octomath::Vector3;

p3d min(p3d const &a, p3d const &b)
{
	return p3d(
		std::min(a.x(), b.x()),
		std::min(a.y(), b.y()),
		std::min(a.z(), b.z()));
}

p3d max(p3d const &a, p3d const &b)
{
	return p3d(
		std::max(a.x(), b.x()),
		std::max(a.y(), b.y()),
		std::max(a.z(), b.z()));
}


bool overlapping(const p3d& b1min, const p3d& b1max, const p3d& b2min, const p3d& b2max)
{
  return (b1max.x() >= b2min.x() && b2max.x() >= b1min.x())
      && (b1max.y() >= b2min.y() && b2max.y() >= b1min.y())
      && (b1max.z() >= b2min.z() && b2max.z() >= b1min.z());
}

// inner function for separating one octomap cell from one robot path segment
halfspace supporting_hyperplane(
	octomap::OcTree::iterator_base const &it,
	poly_sample const &sample,
	float const ellipsoid[3])
{
	// cvxgen
	set_defaults();
	setup_indexing();

	params.Q[0] = sqr(ellipsoid[0]);
	params.Q[1] = sqr(ellipsoid[1]);
	params.Q[2] = sqr(ellipsoid[2]);
	params.Q[3] = 0.0;

	// convert octomap cell into its 8 vertices
	float cx = it.getX();
	float cy = it.getY();
	float cz = it.getZ();
	float hsz = it.getSize() / 2.0;
	int const BOX_N_VTX = 8;
	float cube_pts[BOX_N_VTX][3] = {
		{ cx + hsz, cy + hsz, cz + hsz },
		{ cx + hsz, cy + hsz, cz - hsz },
		{ cx + hsz, cy - hsz, cz + hsz },
		{ cx + hsz, cy - hsz, cz - hsz },
		{ cx - hsz, cy + hsz, cz + hsz },
		{ cx - hsz, cy + hsz, cz - hsz },
		{ cx - hsz, cy - hsz, cz + hsz },
		{ cx - hsz, cy - hsz, cz - hsz },
	};

	#define params_a_ind(i1, i2) params.A[(i1) + SAMPLES_PER_PIECE*2*(i2)]

	// linear constraints for "in" points (robot path)
	for (int dim = 0; dim < 3; ++dim) {
		for (int p = 0; p < SAMPLES_PER_PIECE; ++p) {
			params_a_ind(p, dim) = sample[dim][p];
		}
	}
	for (int i = 0; i < SAMPLES_PER_PIECE; ++i) {
		params_a_ind(i, 3) = 1.0;
	}

	// linear constraints for "out" points (obstacle vertices)
	// repeat cube vertices 4 times to get 32 vs. 32 pt sep hyp
	int const VTX_REP = 4;
	static_assert(VTX_REP * BOX_N_VTX == SAMPLES_PER_PIECE, "oops");
	for (int dim = 0; dim < 3; ++dim) {
		for (int rep = 0; rep < VTX_REP; ++rep) {
			for (int p = 0; p < BOX_N_VTX; ++p) {
				int row = SAMPLES_PER_PIECE + BOX_N_VTX * rep + p;
				assert(row < 2 * SAMPLES_PER_PIECE);
				params_a_ind(row, dim) = -cube_pts[p][dim];
			}
		}
	}
	for (int i = SAMPLES_PER_PIECE; i < 2*SAMPLES_PER_PIECE; ++i) {
		params_a_ind(i, 3) = -1.0;
	}

	set_defaults();
	settings.verbose = 0;
	setup_indexing();
	solve();

	double b0 = vars.beta[0];
	double b1 = vars.beta[1];
	double b2 = vars.beta[2];
	double b3 = vars.beta[3];

	double inv_norm = 1.0 / sqrt(sqr(b0) + sqr(b1) + sqr(b2));
	double a0 = inv_norm * b0;
	double a1 = inv_norm * b1;
	double a2 = inv_norm * b2;
	double bb = -inv_norm * b3;

	// SVM gives us separating hyperplane -
	// make it a supporting hyperplane
	double min_dist = 1000000000;
	for (int i = 0; i < BOX_N_VTX; ++i) {
		double mul = a0 * cube_pts[i][0]
				   + a1 * cube_pts[i][1]
				   + a2 * cube_pts[i][2];
		min_dist = std::min(min_dist, mul - bb);
	}
	bb = bb + min_dist;

	halfspace h;
	h.a[0] = a0;
	h.a[1] = a1;
	h.a[2] = a2;
	h.b = bb;

	return h;
}

polyhedron partition(octomap::OcTree const &ot, poly_sample const &s, float const ellipsoid[3])
{
	// TODO input
	//float const ellipsoid[3] = {0.15, 0.15, 0.15};

	// compute padded bounding box around traj sample
	auto pt = [&s](int i) { return p3d(s[0][i], s[1][i], s[2][i]); };
	p3d bbox_min = pt(0);
	p3d bbox_max = pt(0);
	for (int i = 1; i < SAMPLES_PER_PIECE; ++i) {
		p3d p = pt(i);
		bbox_min = min(bbox_min, p);
		bbox_max = min(bbox_max, p);
	}
	// TODO compute the pad amount intelligently
	float const BBOX_PAD = 1.0; // meters
	bbox_min -= p3d(1, 1, 1) * BBOX_PAD;
	bbox_max += p3d(1, 1, 1) * BBOX_PAD;

	// output
	std::vector<halfspace> part;

	// compute supporting hyperplanes for each occupied cell in bbox
	auto it = ot.begin_leafs_bbx(bbox_min, bbox_max);
	auto end = ot.end_leafs_bbx();
	for (; it != end; ++it) {
		p3d minLeaf = it.getCoordinate() - p3d(it.getSize() / 2, it.getSize() / 2, it.getSize() / 2);
		p3d maxLeaf = it.getCoordinate() + p3d(it.getSize() / 2, it.getSize() / 2, it.getSize() / 2);

		if (overlapping(minLeaf, maxLeaf, bbox_min, bbox_max) && ot.isNodeOccupied(*it)) {
		// double occ = it->getOccupancy();
		// if (occ > OCCUPANCY_THRESHOLD) {
			// it's an occupied cell
			halfspace h = supporting_hyperplane(it, s, ellipsoid);
			part.push_back(h);
		}
	}

	// add half-spaces for the query bbox
	// TODO: can we use octomap data to remove some of these?
	halfspace h_zero;
	h_zero.a[0] = h_zero.a[1] = h_zero.a[2] = 0;
	h_zero.b = 0;
	for (int d = 0; d < 3; ++d) {
		halfspace hmax = h_zero;
		hmax.a[d] = 1;
		hmax.b = bbox_max(d);
		part.push_back(hmax);

		halfspace hmin = h_zero;
		hmin.a[d] = -1;
		hmin.b = -bbox_min(d);
		part.push_back(hmin);
	}

	return part;
}

// evaluate a polynomial using horner's rule.
float polyval(float const p[PP_SIZE], float t)
{
    float x = 0.0;
    for (int i = PP_DEGREE; i >= 0; --i) {
        x = x * t + p[i];
    }
    return x;
}

template <typename T>
T read(std::istream &s)
{
	T x;
	s.read((char *)&x, sizeof(T));
	return x;
}

template <typename T>
void write(std::ostream &s, T const &val)
{
	s.write((char const *)&val, sizeof(T));
}


// argv[1] : octomap file
// argv[2] : .pptrajs file
// argv[3] : output .halfspaces file
int main(int argc, char *argv[])
{
	/*
	.pptrajs format:
	Header:
		N     - int32 - number of robots
		k     - int32 - number of timesteps
		dim   - int32 - dimension
		order - int32 - polynomial degree + 1
	Breaks:
		t - float32[k + 1] - piece break times
	Coefs:
		c - float32[order, dim, k, N] - first dimension contiguous - 
										ascending order (constant term first)
	Sentinel - 0xFFFFFFFF - for sanity check
	*/

	if (argc < 7) {
		std::cerr << "usage: pptraj <octomap_path> <pps_path> <out_path> ellipsoid_rx ellipsoid_ry ellipsoid_rz" << std::endl;
		return 1;
	}

	// TODO: verify throws on error!
	octomap::OcTree ot(argv[1]);

	std::ifstream ppstream(argv[2], std::ios::in | std::ios::binary);
	if (!ppstream.good()) {
		std::cerr << "error loading trajectories from " << argv[2] << std::endl;
		return 1;
	}
	std::ofstream outstream(argv[3], std::ios::out | std::ios::binary);
	if (!outstream.good()) {
		std::cerr << "error opening output file for writing: " << argv[3] << std::endl;
		return 1;
	}

	// read ellipsoid //TODO error checking
	float ellipsoid[3];
	ellipsoid[0] = atof(argv[4]);
	ellipsoid[1] = atof(argv[5]);
	ellipsoid[2] = atof(argv[6]);

	// read the header
	int32_t const N = read<int32_t>(ppstream);
	int32_t const K = read<int32_t>(ppstream);
	int32_t const dim = read<int32_t>(ppstream);
	int32_t const order = read<int32_t>(ppstream);

	assert(dim == 3);
	assert(order == PP_SIZE);

	// read the break times
	std::vector<float> breaks(K + 1);
	for (int i = 0; i < (K + 1); ++i) {
		breaks[i] = read<float>(ppstream);
	}

	// read polynomial coefficients and sample trajectory points
	Array2D<poly_piece> coefs(N, K);
	Array2D<poly_sample> samples(N, K);
	for (int n = 0; n < N; ++n) {
		for (int k = 0; k < K; ++k) {
			for (int d = 0; d < dim; ++d) {
				// collect the coefficients
				for (int i = 0; i < order; ++i) {
					float val = read<float>(ppstream);
					coefs(n,k)[d][i] = val;
				}
				float const *coef_block = &coefs(n,k)[d][0];

				// sample the piece
				float const T_piece = breaks[k + 1] - breaks[k];
				float const dt = T_piece / (SAMPLES_PER_PIECE - 1);
				for (int i = 0; i < SAMPLES_PER_PIECE; ++i) {
					float t = dt * i;
					float val = polyval(coef_block, t);
					samples(n,k)[d][i] = val;
				}
			}
		}
	}

	// check that we got through the whole file
	uint32_t sentinel = read<uint32_t>(ppstream);
	assert(sentinel == 0xFFFFFFFFu);
	ppstream.close();

	// compute all corridors
	Array2D<polyhedron> corridors(N, K);
	std::transform(samples.begin(), samples.end(), corridors.begin(),
		[&ot,&ellipsoid](poly_sample const &s) { return partition(ot, s, ellipsoid); });

	// write the corridors to the file
	/*
	.halfspaces format:
	Header:
		N     - int32 - number of robots
		k     - int32 - number of timesteps
		dim   - int32 - dimension
	for N robots:
		for K timesteps:
			M - int32 - number of halfspaces
			for M halfspaces:
				a - float32[dim] for ax <= b
				b - float32
	Sentinel - 0xFFFFFFFF - for sanity check

	note: M can be different for each robot-timestep pair!
	*/
	write<int32_t>(outstream, N);
	write<int32_t>(outstream, K);
	write<int32_t>(outstream, dim);
	for (int n = 0; n < N; ++n) {
		for (int k = 0; k < K; ++k) {
			auto const &p = corridors(n, k);
			int32_t m = (int32_t)p.size();
			write<int32_t>(outstream, m);
			for (int i = 0; i < m; ++i) {
				halfspace const &h = p[i];
				write<float>(outstream, h.a[0]);
				write<float>(outstream, h.a[1]);
				write<float>(outstream, h.a[2]);
				write<float>(outstream, h.b);
			}
		}
	}
	write<uint32_t>(outstream, 0xFFFFFFFF);
	outstream.close();
}
