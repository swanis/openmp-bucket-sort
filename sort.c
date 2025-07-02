#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <limits.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <float.h>

typedef struct bucket {
	int length;
	int size;
	double arr[];
} bucket_t;

// Assignments
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

// Hoare partition scheme - Wikipedia page
int partition(double *arr, int lo, int hi) {
	double pivot = arr[lo];
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

void bucket_sort(double *nums, int num_elements, int num_buckets, int num_threads) {
	bucket_t *buckets[num_buckets];

	int i;
	double min, max;
	get_min_and_max(nums, num_elements, &min, &max);

	const int initial_bucket_size = num_elements / num_buckets * 1.1;

	#pragma omp parallel num_threads(num_threads)
	{
		int thread_num = omp_get_thread_num();
		int bucket_index;
		bucket_t *bucket;
		for (bucket_index = thread_num; bucket_index < num_buckets; bucket_index += num_threads) {
			bucket = malloc(sizeof(bucket_t) + initial_bucket_size * sizeof(double));
			bucket->size = initial_bucket_size;
			bucket->length = 0;
			buckets[bucket_index] = bucket;
		}

		int i;
		for (i = 0; i < num_elements; i++) {
			bucket_index = nums[i] != min ? ceil((nums[i] - min) / (max - min) * num_buckets) - 1 : 0;

			if (bucket_index % num_threads == thread_num) {
				bucket = buckets[bucket_index];

				if (++bucket->length > bucket->size) {
					bucket->size = bucket->size * 1.5 + 1;
					bucket = realloc(bucket, sizeof(bucket_t) + bucket->size * sizeof(double));
					buckets[bucket_index] = bucket;
				}

				bucket->arr[bucket->length - 1] = nums[i];
			}
		}

		for (bucket_index = thread_num; bucket_index < num_buckets; bucket_index += num_threads) {
			bucket = buckets[bucket_index];
			quick_sort(bucket->arr, 0, bucket->length - 1);
		}

		// We block until all threads have sorted their buckets
		#pragma omp barrier

		// We write back the elements in our buckets to the global array at their appropriate places
		int base = 0;
		int j;
		for (i = 0; i < num_buckets; i++) {
			if (i % num_threads == thread_num) {
				bucket = buckets[i];

				for (j = 0; j < bucket->length; j++) {
					nums[base+j] = bucket->arr[j];
				}
			}

			base += buckets[i]->length;
		}
	}

	/*for (i = 0; i < num_elements; i++) {
		printf("%7.3f ", nums[i]);
	}

	printf("\n");*/

	for (i = 0; i < num_buckets; i++) {
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

// from https://stackoverflow.com/a/33059025
double randfrom(double min, double max) {
	double range = (max - min);
	double div = RAND_MAX / range;
	return min + (rand() / div);
}

void generate_uniform_nums(double *nums, int num_elements) {
	int i;
	for (i = 0; i < num_elements; i++) {
		nums[i] = randfrom(0, 1000000);
	}
}

// Box-Muller transform (https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
void generate_normal_nums(double *nums, int num_elements) {
	double u1, u2, z0, value;
	int i;
	for (i = 0; i < num_elements; i++) {
		u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
		u2 = (rand() + 1.0) / (RAND_MAX + 2.0);

		z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

		value = z0 * 166667 + 500000;
		nums[i] = value;
	}
}

// from https://stackoverflow.com/a/34558404
double ran_expo(double lambda){
	double u;
	u = rand() / (RAND_MAX + 1.0);
	return -log(1- u) / lambda;
}

void generate_exponential_nums(double *nums, int num_elements) {
	int i;
	for (i = 0; i < num_elements; i++) {
		nums[i] = ran_expo(0.000003);
	}
}

int main(int argc, char *argv[]) {
	if (argc != 4) {
		printf("./sort num_elements num_buckets num_threads\n");
		return 1;
	}

	int num_elements = atoi(argv[1]);
	int num_buckets = atoi(argv[2]);
	int num_threads = atoi(argv[3]);

	srand(time(NULL)); // Seed according to current time

	double *nums = malloc(sizeof(double) * num_elements); // go with heap allocation, it is safest for runtime-sized arrays

	generate_uniform_nums(nums, num_elements);
	double time = get_wall_seconds();
	bucket_sort(nums, num_elements, num_buckets, num_threads);
	printf("Sorting uniform numbers took %7.3f wall seconds\n", get_wall_seconds()-time);
	printf("Properly sorted: %i\n", ensure_sorted(nums, num_elements));

	generate_normal_nums(nums, num_elements);
	time = get_wall_seconds();
	bucket_sort(nums, num_elements, num_buckets, num_threads);
	printf("Sorting normal numbers took %7.3f wall seconds\n", get_wall_seconds()-time);
	printf("Properly sorted: %i\n", ensure_sorted(nums, num_elements));

	generate_exponential_nums(nums, num_elements);
	time = get_wall_seconds();
	bucket_sort(nums, num_elements, num_buckets, num_threads);
	printf("Sorting exponential numbers took %7.3f wall seconds\n", get_wall_seconds()-time);
	printf("Properly sorted: %i\n", ensure_sorted(nums, num_elements));

	free(nums);

	return 0;
}
