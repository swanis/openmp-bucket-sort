#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <limits.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <float.h>

#define NUM_THREADS 23
#define NUM_ELEMENTS 100000000
#define NUM_BUCKETS 10000

typedef struct bucket {
	int length;
	int size;
	double arr[];
} bucket_t;

static double get_wall_seconds() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000;
	return seconds;
}

void get_min_and_max(double *arr, int len, double *min, double *max) {
	double local_min = DBL_MAX;
	double local_max = -DBL_MAX;
	double curr;
	int i;
	for (i = 0; i < len; i++) {
		curr = arr[i];
		if (curr < local_min) {
			local_min = curr;
		}
		if (curr > local_max) {
			local_max = curr;
		}
	}
	*min = local_min;
	*max = local_max;
}

void bubble_sort(bucket_t *bucket) {
	int i;
	double i_value;
	for (i = 0; i < bucket->length; i++) {
		i_value = bucket->arr[i];
		int j;
		double j_value;
		for (j = i+1; j < bucket->length; j++) {
			j_value = bucket->arr[j];
			if (j_value < i_value) {
				bucket->arr[i] = j_value;
				bucket->arr[j] = i_value;
				i_value = j_value;
			}
		}
	}
}

// Hoare partition scheme
int partition(double *arr, int lo, int hi) {
	double pivot = arr[lo]; // choose pivot better
	int i = lo - 1, j = hi + 1;
	double tmp;

	while (1) {
		do {
			i += 1;
		} while (arr[i] < pivot);

		do {
			j -= 1;
		} while (arr[j] > pivot);

		if (i >= j) {
			return j;
		}

		tmp = arr[i];
		arr[i] = arr[j];
		arr[j] = tmp;
	}
}

void quick_sort(double *arr, int lo, int hi) {
	if (lo >= 0 && hi >= 0 && lo < hi) {
		int pivot_index = partition(arr, lo, hi);
		quick_sort(arr, lo, pivot_index);
		quick_sort(arr, pivot_index + 1, hi);
	}
}

// from https://stackoverflow.com/a/33059025
double randfrom(double min, double max) {
	double range = (max - min);
	double div = RAND_MAX / range;
	return min + (rand() / div);
}

void generate_uniform_nums(double *nums) {
	int i;
	for (i = 0; i < NUM_ELEMENTS; i++) {
		nums[i] = randfrom(0, 100);
	}
}

// Box-Muller transform (https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
void generate_normal_nums(double *nums) {
	double u1, u2, z0, value;
	int i;
	for (i = 0; i < NUM_ELEMENTS; i++) {
		u1 = (rand() + 1.0) / (RAND_MAX + 2.0); //maybe look into this random number generation - the one I posted on discord uses + 1.0 instead of 2.0 in the denominator
		u2 = (rand() + 1.0) / (RAND_MAX + 2.0);

		z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

		value = z0 * 15 + 50;
		nums[i] = value;
	}
}

// from https://stackoverflow.com/a/34558404
double ran_expo(double lambda){
	double u;
	u = rand() / (RAND_MAX + 1.0);
	return -log(1- u) / lambda;
}

void generate_exponential_nums(double *nums) {
	int i;
	for (i = 0; i < NUM_ELEMENTS; i++) {
		nums[i] = ran_expo(0.03);
	}
}

void bucket_sort(double *nums) {
	bucket_t *buckets[NUM_BUCKETS];

	int i;
	double min, max;
	get_min_and_max(nums, NUM_ELEMENTS, &min, &max);

	// We should only time the concurrent sorting, using the time function from previous labs etc

	#pragma omp parallel num_threads(NUM_THREADS)
	{
		int thread_num = omp_get_thread_num();
		int bucket_index, bucket_size;
		for (bucket_index = thread_num; bucket_index < NUM_BUCKETS; bucket_index+=NUM_THREADS) {
			bucket_size = NUM_ELEMENTS / NUM_BUCKETS * 1.1;
			bucket_t *bucket = malloc(sizeof(bucket_t) + bucket_size * sizeof(double));
			bucket->size = bucket_size;
			bucket->length = 0;
			buckets[bucket_index] = bucket;
		}

		int i, matching_bucket;
		for (i = 0; i < NUM_ELEMENTS; i++) {
			// second part is for ceiling, so add 1 if there is a remainder. and -1 at the end is included in the formula
			matching_bucket = nums[i] != min ? ceil((nums[i]-min)/(max-min)*NUM_BUCKETS)-1 : 0;
			if (matching_bucket % NUM_THREADS == thread_num) {
				bucket_t *bucket = buckets[matching_bucket];
				if (++bucket->length > bucket->size) {
					//realloc - maybe add more than 1 to bucket size
					bucket_size += 1;
					bucket = realloc(bucket, sizeof(bucket_t) + bucket_size * sizeof(double));
					bucket->size = bucket_size;
					buckets[matching_bucket] = bucket;
				}

				bucket->arr[bucket->length - 1] = nums[i];
			}
		}

		for (bucket_index = thread_num; bucket_index < NUM_BUCKETS; bucket_index+=NUM_THREADS) {
			//bubble_sort(bucket);
			bucket_t *bucket = buckets[bucket_index];
			quick_sort(bucket->arr, 0, bucket->length - 1);
		}

		// BLOCK UNTIL ALL THREADS HAVE SORTED
		#pragma omp barrier

		// WRITE BACK TO GLOBAL ARRAY (ADD LENGTH OF ALL BUCKETS UNDER CURRENT ONE TO GET LOCATION TO WRITE TO)
		int base = 0;
		int j;
		for (i = 0; i < NUM_BUCKETS; i++) {
			if (i % NUM_THREADS == thread_num) {
				bucket_t *bucket = buckets[i];
				for (j = 0; j < bucket->length; j++) {
					nums[base+j] = bucket->arr[j];
				}
			}
			base += buckets[i]->length;
		}
	}

	/*for (i = 0; i < NUM_ELEMENTS; i++) {
		printf("%7.3f ", nums[i]);
	}

	printf("\n");*/

	for (i = 0; i < NUM_BUCKETS; i++) {
		free(buckets[i]);
	}
}

int ensure_sorted(double *arr, int len) {
	double curr = -DBL_MAX;
	int i;
	for (i = 0; i < len; i++) {
		if (arr[i] < curr) {
			return 0;
		}
		curr = arr[i];
	}
	return 1;
}

int main() {
	srand(time(NULL)); // Seed according to current time

	double *nums = malloc(sizeof(double) * NUM_ELEMENTS); // go with heap allocation, it is safest for runtime-sized arrays

	generate_uniform_nums(nums);
	double time = get_wall_seconds();
	bucket_sort(nums);
	// maybe ensure it is sorted
	printf("Sorting uniform numbers took %7.3f wall seconds\n", get_wall_seconds()-time);
	printf("Properly sorted: %i\n", ensure_sorted(nums, NUM_ELEMENTS));

	generate_normal_nums(nums);
	time = get_wall_seconds();
	bucket_sort(nums);
	printf("Sorting normal numbers took %7.3f wall seconds\n", get_wall_seconds()-time);
	printf("Properly sorted: %i\n", ensure_sorted(nums, NUM_ELEMENTS));

	generate_exponential_nums(nums);
	time = get_wall_seconds();
	bucket_sort(nums);
	// maybe ensure it is sorted
	printf("Sorting exponential numbers took %7.3f wall seconds\n", get_wall_seconds()-time);
	printf("Properly sorted: %i\n", ensure_sorted(nums, NUM_ELEMENTS));

	free(nums);
}
