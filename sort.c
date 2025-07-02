#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <limits.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <float.h>

// We define our bucket struct as such
typedef struct bucket {
	int length;
	int size;
	double arr[];
} bucket_t;

// Function used for getting the current time
// Provided during the labs
static double get_wall_seconds() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000;
	return seconds;
}

// Function for getting the minimum and maximum elements in arr
// When it has obtained the values, it will write them to the doubles that min and max point to
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

// Partitions arr such that all elements smaller than a chosen pivot element
// are in one partition and the ones bigger than it are in another partition
// Returns the index of the pivot element
// This follows the Hoare partition scheme, and is based on pseudo code from https://en.wikipedia.org/w/index.php?title=Quicksort&oldid=1293231937
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

// Sorts arr from index lo to hi
// Based on pseudo code from https://en.wikipedia.org/w/index.php?title=Quicksort&oldid=1293231937
void quick_sort(double *arr, int lo, int hi) {
	if (lo >= 0 && hi >= 0 && lo < hi) {
		int pivot_index = partition(arr, lo, hi);
		quick_sort(arr, lo, pivot_index);
		quick_sort(arr, pivot_index + 1, hi);
	}
}

// Sorts nums using our bucket sort implementation, assuming it contains num_elements elements
// It uses num_buckets buckets and will be parallelized using num_threads threads assuming num_threads > 1
void bucket_sort(double *nums, int num_elements, int num_buckets, int num_threads) {
	// Defines an array of pointers to num_buckets buckets
	bucket_t *buckets[num_buckets];

	// Gets the minimum and maximum values in arr. After the get_min_and_max function
	// completes, the values will be written to min and max
	double min, max;
	get_min_and_max(nums, num_elements, &min, &max);

	// Defines the initial bucket size
	const int initial_bucket_size = num_elements / num_buckets * 1.1;

	// We create num_threads threads
	#pragma omp parallel num_threads(num_threads)
	{
		int thread_num = omp_get_thread_num();

		// The thread creates its own buckets. Buckets are assigned in a static round robin manner among the threads
		int bucket_index;
		bucket_t *bucket;
		for (bucket_index = thread_num; bucket_index < num_buckets; bucket_index += num_threads) {
			bucket = malloc(sizeof(bucket_t) + initial_bucket_size * sizeof(double));
			bucket->size = initial_bucket_size;
			bucket->length = 0;
			buckets[bucket_index] = bucket;
		}

		// The thread goes over all elements to figure out which ones its own buckets should contain
		int i;
		for (i = 0; i < num_elements; i++) {
			// Retrieves the index of the bucket for the current element
			bucket_index = nums[i] != min ? ceil((nums[i] - min) / (max - min) * num_buckets) - 1 : 0;

			// If the index of the bucket for the current element is one of the threads own buckets, it wants to add the element to the bucket
			if (bucket_index % num_threads == thread_num) {
				bucket = buckets[bucket_index];

				// If there is no space for the element in the bucket, increase the size of the bucket by reallocating the entire bucket
				if (++bucket->length > bucket->size) {
					// We increase the size in an exponential manner. We always add 1 in case the bucket size is currently 0
					bucket->size = bucket->size * 1.5 + 1;
					bucket = realloc(bucket, sizeof(bucket_t) + bucket->size * sizeof(double));
					buckets[bucket_index] = bucket;
				}

				// Add the element to the bucket
				bucket->arr[bucket->length - 1] = nums[i];
			}
		}

		// We go over our buckets and sort them using quick sort
		for (bucket_index = thread_num; bucket_index < num_buckets; bucket_index += num_threads) {
			bucket = buckets[bucket_index];
			quick_sort(bucket->arr, 0, bucket->length - 1);
		}

		// We block until all threads have sorted their buckets
		#pragma omp barrier

		// We write back the elements in our buckets to the global array at
		// their appropriate places by summing the lengths of all buckets before them
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

	// We free the memory allocated for every bucket
	int i;
	for (i = 0; i < num_buckets; i++) {
		free(buckets[i]);
	}
}

// Ensures that arr is sorted, assuming it contains len elements
// Returns 0 if it is not sorted and 1 if it is
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

// Generates a random floating point number from min to max
// Code taken from https://stackoverflow.com/a/33059025
double randfrom(double min, double max) {
	double range = (max - min);
	double div = RAND_MAX / range;
	return min + (rand() / div);
}

// Fills nums with num_elements random numbers from 0 to 1000000 following a uniform distribution
void generate_uniform_nums(double *nums, int num_elements) {
	int i;
	for (i = 0; i < num_elements; i++) {
		nums[i] = randfrom(0, 1000000);
	}
}

// Fills nums with num_elements random numbers following a normal distribution with mean == 500000 and standard deviance == 166667
// The Box-Muller transform method was used, and the following
// Wikipediga page was consulted: https://en.wikipedia.org/w/index.php?title=Box%E2%80%93Muller_transform&oldid=1294410019
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

// Generates a random number following an exponential distribution with lambda value lambda
// Code taken from https://stackoverflow.com/a/34558404
double ran_expo(double lambda) {
	double u;
	u = rand() / (RAND_MAX + 1.0);
	return -log(1- u) / lambda;
}

// Fills nums with num_elements random numbers following an exponential distribution with lambda 0.000003
void generate_exponential_nums(double *nums, int num_elements) {
	int i;
	for (i = 0; i < num_elements; i++) {
		nums[i] = ran_expo(0.000003);
	}
}

int main(int argc, char *argv[]) {
	// We require exactly three command line arguments
	if (argc != 4) {
		printf("./sort num_elements num_buckets num_threads\n");
		return 1;
	}

	// We parse the command line arguments
	int num_elements = atoi(argv[1]);
	int num_buckets = atoi(argv[2]);
	int num_threads = atoi(argv[3]);

	// We seed according to current time
	srand(time(NULL));

	// We allocate memory for num_elements of type double on the heap
	double *nums = malloc(sizeof(double) * num_elements);

	// We fill nums with num_elements random numbers following a uniform distribution and
	// measure the time it takes to sort it using our bucket sort implementation. We also
	// ensure the array ends up correctly sorted
	generate_uniform_nums(nums, num_elements);
	double time = get_wall_seconds();
	bucket_sort(nums, num_elements, num_buckets, num_threads);
	printf("Sorting uniform numbers took %7.3f wall seconds\n", get_wall_seconds()-time);
	printf("Properly sorted: %i\n", ensure_sorted(nums, num_elements));

	// We fill nums with num_elements random numbers following a normal distribution and
	// measure the time it takes to sort it using our bucket sort implementation. We also
	// ensure the array ends up correctly sorted
	generate_normal_nums(nums, num_elements);
	time = get_wall_seconds();
	bucket_sort(nums, num_elements, num_buckets, num_threads);
	printf("Sorting normal numbers took %7.3f wall seconds\n", get_wall_seconds()-time);
	printf("Properly sorted: %i\n", ensure_sorted(nums, num_elements));

	// We fill nums with num_elements random numbers following an exponential distribution and
	// measure the time it takes to sort it using our bucket sort implementation. We also
	// ensure the array ends up correctly sorted
	generate_exponential_nums(nums, num_elements);
	time = get_wall_seconds();
	bucket_sort(nums, num_elements, num_buckets, num_threads);
	printf("Sorting exponential numbers took %7.3f wall seconds\n", get_wall_seconds()-time);
	printf("Properly sorted: %i\n", ensure_sorted(nums, num_elements));

	// We free the memory we allocated for nums
	free(nums);

	return 0;
}
