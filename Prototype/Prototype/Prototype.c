
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include "prototype_header.h"
#include "Parallel.h"

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
train_network(double* network, int* image) {
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
			//note that this stores the correlations of every i to one j continuously--e.g., all the relations TO
			//the first pixel are stored from 0 to 34.
		}
	}
}

void
draw_pattern(int* pattern) {
	system("cls"); //only works on Windows
	for (int i = 0; i < PATTERN_SIZE; i++) {
		if (i % 5 == 0)
			printf("\n");
		if (*(pattern + i) == 1)
			printf("#");
		else
			printf(".");
	}
}

void
generate_cue(int* pattern, int* cue_pattern, int noise_threshold) {
	for (int i = 0; i < PATTERN_SIZE; i++) {
		cue_pattern[i] = pattern[i];
	}
	for (int i = 0; i < PATTERN_SIZE; i++) {
		//flip pixel according to noise
		if (rand() % 101 < noise_threshold)
			cue_pattern[i] = 1 - cue_pattern[i];
	}
}

int
neuron_out(double* network, int* pattern, int neuron_in) {
	double sum_weights = 0;
	int pixel_value = 0;
	for (int i = 0; i < PATTERN_SIZE; i++) {
		sum_weights += *(pattern + i) * *(network + neuron_in + i*PATTERN_SIZE); 
		//multiply the value of the ith pixel by the correlation of i and the current pixel
		//add to the weight. The sign of the weight shows wheteher the pixel should be on.
	}
	if (sum_weights > 0)
		pixel_value = 1;
	else
		pixel_value = -1;
	return pixel_value;
}

void
shuffle(int *array, size_t n) {
	if (n > 1)
	{
		size_t i;
		for (i = 0; i < n - 1; i++)
		{
			size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
			int t = array[j];
			array[j] = array[i];
			array[i] = t;
		}
	}
}

void
recall_step(double* network, int* pattern, int* sequence, int step) {
	if(step == 0)
		shuffle(sequence, sizeof(sequence)); //randomise sequence order
	//rather than consistently evaluating from the first pixel
	//then update the image based on the network
	int new_pixel;
	new_pixel = neuron_out(network, pattern, sequence[step]); //see whether the network's
	//correlations suggest a pixel should be different.
	if (new_pixel != pattern[sequence[step]])
		pattern[sequence[step]] = new_pixel;
	draw_pattern(pattern);
}

void
recall(double* network, int* pattern) {
	int sequence[PATTERN_SIZE];
	for (int i = 0; i < PATTERN_SIZE; i++) {
		sequence[i] = i;
	}
	int step = 0;
	bool exit = false;
	do	{
		recall_step(network, pattern, sequence, step);
		step++;
		if (step == PATTERN_SIZE) {
			step = 0; //causes the next recall step to shuffle the order once more.
			char option;
			printf("\n Enter Q to quit or anything else to continue.");
			Sleep(500);
			scanf("%c", &option);
			option = toupper(option);
			if (option == 'Q')
				exit = true;
		}
		Sleep(100); 
	} while (exit == false);
}

int main(void) {
	//start RNG
	srand(time(NULL));

	//array pattern for the letter A
	int a_pattern[] = { 0, 0, 1, 0, 0,
						0, 1, 0, 1, 0,
						1, 0, 0, 0, 1,
						1, 1, 1, 1, 1,
						1, 0, 0, 0, 1,
						1, 0, 0, 0, 1,
						1, 0, 0, 0, 1 };

	//array pattern for the letter s
	int s_pattern[] = { 1, 1, 1, 1, 1,
						1, 0, 0, 0, 0,
						0, 1, 0, 0, 0,
						0, 0, 1, 0, 0,
						0, 0, 0, 1, 0,
						0, 0, 0, 0, 1,
						1, 1, 1, 1, 1 };

	//array pattern for the letter t
	int t_pattern[] = { 1, 1, 1, 1, 1,
						0, 0, 1, 0, 0,
						0, 0, 1, 0, 0,
						0, 0, 1, 0, 0,
						0, 0, 1, 0, 0,
						0, 0, 1, 0, 0,
						0, 0, 1, 0, 0 };

	//array pattern for the letter u
	int u_pattern[] = { 1, 0, 0, 0, 1,
						1, 0, 0, 0, 1,
						1, 0, 0, 0, 1,
						1, 0, 0, 0, 1,
						1, 0, 0, 0, 1,
						1, 0, 0, 0, 1,
						1, 1, 1, 1, 1 };

	//create a blank network
	double weighted_network[NETWORK_SIZE];
	for (int i = 0; i < NETWORK_SIZE; i++) {
		weighted_network[i] = 0;
	}

	train_network(weighted_network, a_pattern);
	train_network(weighted_network, s_pattern);
	train_network(weighted_network, t_pattern);
	train_network(weighted_network, u_pattern);
	int cue[PATTERN_SIZE]; 

	bool exit = false;
	do {
		bool generated = false;

		char gpu;
		char option;
		int noise;
		printf("Enter G to use the GPU instead");
		scanf("%c", &gpu);
		gpu = toupper(gpu);
		printf("Enter a character: A, S, T, U to recall that letter, Q to quit. \n");
		scanf("%c", &option);
		option = toupper(option);
		if (option != 'Q') {
			printf("Please enter a value from 0-100 for noise. \n");
			scanf("%d", &noise);
		}
		switch (option) {
		case 'A':
			generate_cue(a_pattern, cue, noise);
			generated = true;
			break;
		case 'S':
			generate_cue(s_pattern, cue, noise);
			generated = true;
			break;
		case 'T':
			generate_cue(t_pattern, cue, noise);
			generated = true;
			break;
		case 'U':
			generate_cue(u_pattern, cue, noise);
			generated = true;
			break;
		case 'Q':
			exit = true;
		}
		if (generated) {
			negative_image(cue);
			draw_pattern(cue);
			if (gpu == 'G')
				parallel_recall(weighted_network, cue);
			else
				recall(weighted_network, cue);
		}
	} while (exit == false);

	return 0;
}