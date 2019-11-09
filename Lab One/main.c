#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>

// Returns a random value between -1 and 1
double getRand(unsigned int* seed)
{
	return (double)rand_r(seed) * 2 / (double)(RAND_MAX)-1;
}

int Count_Number_Of_Samples_In_Circle(long number_of_tosses, int seed)
{
	double x, y;
	int numberOfSamplesInCircle = 0;

	for (int i = 0; i < number_of_tosses; i++)
	{
		x = getRand(&seed);
		y = getRand(&seed);
		if (x * x + y * y < 1) {
			numberOfSamplesInCircle++;
		}
	}

	return numberOfSamplesInCircle;
}

long double Calculate_Pi_Sequential(long long number_of_tosses) {
	unsigned int seed = (unsigned int)time(NULL);

	//Calls Count_Number_Of_Samples_In_Circle without dividing the workload
	int count = Count_Number_Of_Samples_In_Circle(number_of_tosses, seed);

	return (double)count / number_of_tosses * 4;
}

long double Calculate_Pi_Parallel(long long number_of_tosses)
{
	int numberOfThreads = omp_get_max_threads();
	int workloadPerThread = number_of_tosses / numberOfThreads;

	int numberOfSamplesInCircle = 0;

	//Splits the workload by the number of threads
	#pragma omp parallel for num_threads(numberOfThreads)
	for (int i = 0; i < numberOfThreads; i++)
	{
		unsigned int seed = (unsigned int)time(NULL) + (unsigned int)omp_get_thread_num();

		//Calls Count_Number_Of_Samples_In_Circle one per thread with the divided workload
		numberOfSamplesInCircle += Count_Number_Of_Samples_In_Circle(workloadPerThread, seed);
	}

	return (double)numberOfSamplesInCircle / number_of_tosses * 4;
}

int main() {
	struct timeval start, end;

	long long num_tosses = 10000000;

	printf("Timing sequential...\n");
	gettimeofday(&start, NULL);
	long double sequential_pi = Calculate_Pi_Sequential(num_tosses);
	gettimeofday(&end, NULL);
	printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double)(end.tv_usec - start.tv_usec) / 1000000);

	printf("Timing parallel...\n");
	gettimeofday(&start, NULL);
	long double parallel_pi = Calculate_Pi_Parallel(num_tosses);
	gettimeofday(&end, NULL);
	printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double)(end.tv_usec - start.tv_usec) / 1000000);

	// This will print the result to 10 decimal places
	printf("π = %.10Lf (sequential)\n", sequential_pi);
	printf("π = %.10Lf (parallel)", parallel_pi);

	return 0;
}