#include <string.h> // memset, memcpy
#include <math.h> //tanhf

#include "nn.h"

// unconventional: include generated c-file in here
#include "generated_weights.c"

static float temp1[32];
static float temp2[32];

static float deepset_sum_neighbor[8];
static float deepset_sum_obstacle[8];

static float closest_dist;
static float closest[2];
static float min_distance;

static const float b_gamma = 0.1;
static const float b_exph = 1.0;
static const float robot_radius = 0.15; // m
static const float max_v = 0.5; // m/s


static float relu(float num) {
	if (num > 0) {
		return num;
	} else {
		return 0;
	}
}

static void layer(int rows, int cols, const float in[], const float layer_weight[][cols], const float layer_bias[],
			float * output, int use_activation) {
	for(int ii = 0; ii < cols; ii++) {
		output[ii] = 0;
		for (int jj = 0; jj < rows; jj++) {
			output[ii] += in[jj] * layer_weight[jj][ii];
		}
		output[ii] += layer_bias[ii];
		if (use_activation == 1) {
			output[ii] = relu(output[ii]);
		}
	}
}

static const float* n_phi(const float input[]) {
	layer(2, 32, input, weights_n_phi.layers_0_weight, weights_n_phi.layers_0_bias, temp1, 1);
	layer(32, 32, temp1, weights_n_phi.layers_1_weight, weights_n_phi.layers_1_bias, temp2, 1);
	layer(32, 8, temp2, weights_n_phi.layers_2_weight, weights_n_phi.layers_2_bias, temp1, 0);

	return temp1;
}

static const float* n_rho(const float input[]) {
	layer(8, 32, input, weights_n_rho.layers_0_weight, weights_n_rho.layers_0_bias, temp1, 1);
	layer(32, 32, temp1, weights_n_rho.layers_1_weight, weights_n_rho.layers_1_bias, temp2, 1);
	layer(32, 8, temp2, weights_n_rho.layers_2_weight, weights_n_rho.layers_2_bias, temp1, 0);

	return temp1;
}

static const float* o_phi(const float input[]) {
	layer(2, 32, input, weights_o_phi.layers_0_weight, weights_o_phi.layers_0_bias, temp1, 1);
	layer(32, 32, temp1, weights_o_phi.layers_1_weight, weights_o_phi.layers_1_bias, temp2, 1);
	layer(32, 8, temp2, weights_o_phi.layers_2_weight, weights_o_phi.layers_2_bias, temp1, 0);

	return temp1;
}

static const float* o_rho(const float input[]) {
	layer(8, 32, input, weights_o_rho.layers_0_weight, weights_o_rho.layers_0_bias, temp1, 1);
	layer(32, 32, temp1, weights_o_rho.layers_1_weight, weights_o_rho.layers_1_bias, temp2, 1);
	layer(32, 8, temp2, weights_o_rho.layers_2_weight, weights_o_rho.layers_2_bias, temp1, 0);

	return temp1;
}

static const float* psi(const float input[]) {
	layer(18, 32, input, weights_psi.layers_0_weight, weights_psi.layers_0_bias, temp1, 1);
	layer(32, 32, temp1, weights_psi.layers_1_weight, weights_psi.layers_1_bias, temp2, 1);
	layer(32, 2, temp2, weights_psi.layers_2_weight, weights_psi.layers_2_bias, temp1, 0);

	// // scaling part
	// const float psi_min = -0.5;
	// const float psi_max = 0.5;
	// for (int i = 0; i < 2; ++i) {
	// 	temp1[i] = (tanhf(temp1[i]) + 1.0f) / 2.0f * ((psi_max - psi_min) + psi_min);
	// }

	return temp1;
}

static float clip(float value, float min, float max) {
	if (value < min) {
		return min;
	}
	if (value > max) {
		return max;
	}
	return value;
}

static void barrier(float x, float y, float D, float* vx, float *vy) {
	float normP = sqrtf(x*x + y*y);
	float H = normP - D;

	float factor = powf(H, -b_exph) / normP;

	*vx = b_gamma * factor * x;
	*vy = b_gamma * factor * y;
}

static void APF(float* vel)
{
	if (isfinite(closest_dist)) {
		float vx, vy;
		barrier(closest[0], closest[1], min_distance, &vx, &vy);
		vel[0] -= vx;
		vel[1] -= vy;
	}
}

void nn_reset()
{
	memset(deepset_sum_neighbor, 0, sizeof(deepset_sum_neighbor));
	memset(deepset_sum_obstacle, 0, sizeof(deepset_sum_obstacle));

	closest_dist = INFINITY;
}

void nn_add_neighbor(const float input[2])
{
	const float* phi = n_phi(input);
	// sum result
	for (int i = 0; i < 8; ++i) {
		deepset_sum_neighbor[i] += phi[i];
	}

	float dist = fmaxf(sqrtf(powf(input[0], 2) + powf(input[1], 2)) - robot_radius, 0);
	if (dist < closest_dist) {
		memcpy(closest, input, sizeof(closest));
		closest_dist = dist;
		min_distance = 2 * robot_radius;
	}
}

void nn_add_obstacle(const float input[2])
{
	const float* phi = o_phi(input);
	// sum result
	for (int i = 0; i < 8; ++i) {
		deepset_sum_obstacle[i] += phi[i];
	}

	float closest_x = clip(0, input[0] - 0.5f, input[0] + 0.5f);
	float closest_y = clip(0, input[1] - 0.5f, input[1] + 0.5f);
	float dist = sqrtf(powf(closest_x, 2) + powf(closest_y, 2));
	if (dist < closest_dist) {
		closest[0] = closest_x;
		closest[1] = closest_y;
		closest_dist = dist;
		min_distance = robot_radius;
	}
}

const float* nn_eval(const float goal[2])
{
	static float pi_input[18];

	const float* neighbors = n_rho(deepset_sum_neighbor);
	memcpy(&pi_input[0], neighbors, 8 * sizeof(float));

	const float* obstacles = o_rho(deepset_sum_obstacle);
	memcpy(&pi_input[8], obstacles, 8 * sizeof(float));

	memcpy(&pi_input[16], goal, 2 * sizeof(float));

	const float* empty = psi(pi_input);
	temp1[0] = empty[0];
	temp1[1] = empty[1];

	// APF(temp1);

	// float inv_alpha = fmaxf(fabsf(temp1[0]), fabsf(temp1[1])) / max_v;
	// inv_alpha = fmaxf(inv_alpha, 1.0);
	// temp1[0] /= inv_alpha;
	// temp1[1] /= inv_alpha;

	return temp1;
}

