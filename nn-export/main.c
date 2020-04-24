#include <stdio.h>

#include "nn.h"

int main()
{

	nn_reset();

	const float neighbor1[] = {1,2,3,4};
	nn_add_neighbor(neighbor1);
	const float neighbor2[] = {5,6,7,8};
	nn_add_neighbor(neighbor2);

	const float obstacle1[] = {1,2};
	nn_add_obstacle(obstacle1);

	float goal[4] = {0,0,3,4};

	const float* result = nn_eval(goal);

	for (int i = 0; i < 2; ++i) {
		printf("%f ", result[i]);
	}
	printf("\n");

	return 0;
}