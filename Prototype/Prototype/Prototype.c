
#include <stdio.h>

#define IMAGE_WIDTH		5
#define IMAGE_HEIGHT	7
#define PATTERN_SIZE	IMAGE_WIDTH*IMAGE_HEIGHT
#define NETWORK_SIZE	PATTERN_SIZE*PATTERN_SIZE

void
negative_image(int* image) {
	for (int i = 0; i < PATTERN_SIZE; i++) {
		//convert 0's to negatives
		*(image + i) = *(image + i) * 2 - 1; //1*2 = 2, 0*2 = 0
	}
}

void 
train_network(int* network, int* image) {
	//converts the image to be learned into a [-1, 1] format
	//new array to not change the original pattern
	int transformed_image[PATTERN_SIZE];
	for (int i = 0; i < PATTERN_SIZE; i++) {
		transformed_image[i] = *(image + i);
	}
	negative_image(transformed_image);

	for (int i = 0; i < PATTERN_SIZE; i++) { //relate every pixel
		for (int j = 0; j < PATTERN_SIZE; j++) { //to every other pixel, including itself 
			*(network + i + j*PATTERN_SIZE) += transformed_image[i] * transformed_image[j];
		}
	}
}

void
draw_pattern(int* pattern) {
	for (int i = 0; i < PATTERN_SIZE; i++) {
		if (i % 5 == 0)
			printf("\n");
		if (*(pattern + i) == 0)
			printf(".");
		else
			printf("#");
	}
}

int main(void) {

	//array pattern for the letter A
	int a_pattern[] = { 0, 0, 1, 0, 0,
						0, 1, 0, 1, 0,
						1, 0, 0, 0, 1,
						1, 1, 1, 1, 1,
						1, 0, 0, 0, 1,
						1, 0, 0, 0, 1,
						1, 0, 0, 0, 1 };

	//create a blank network
	double weighted_network[NETWORK_SIZE];
	for (int i = 0; i < NETWORK_SIZE; i++) {
		weighted_network[i] = 0;
	}

	draw_pattern(a_pattern);

	return 0;
}