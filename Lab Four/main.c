/// Program was adapted from https://github.com/eduardlopez/quicksort-parallel
/// The sequential version is largely the same but the parallel version was not used
/// Instead I adapted the descriptions from https://www.cs.wcupa.edu/rkline/ds/fast-sorts.html
/// Which did a great job explaining the reason some sorting algorithms need optimising in the form of
/// a Cut Off where automatic thread queuing should stop and instead be handled by a single thread

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include <memory.h>

// Number of values to sort
static const long Num_To_Sort = 1000000;

// The number of sorts left where it becomes
// faster to have a single thread create tasks
static const long Cut_Off = 10000;

// Sequential version of QuickSort
void QuickSort(int* arr, int left, int right)
{
    int i = left;
    int j = right;
    int tmp;

    // A pivot element used to split the array into two parts
    const int pivot = arr[(left + right) / 2];

    // Partitions and swaps the elements from the ends down to the center
    while (i <= j)
    {
        while (arr[i] < pivot)
        {
            i++;
        }
        while (arr[j] > pivot)
        {
            j--;
        }
        if (i <= j)
        {
            tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            i++;
            j--;
        }
    }

    // Recursively calls QuickSort until entire
    // array has been iterated through
    if (left < j)
    {
        QuickSort(arr, left, j);
    }
    if (i < right)
    {
        QuickSort(arr, i, right);
    }
}

// Calls sequential version of Quick Sort
void sort_s(int* arr) {

    const int leftStart = 0;
    const int rightStart = Num_To_Sort - 1;

    QuickSort(arr, leftStart, rightStart);
}

/// A typical implementation of QuickSort with the intention to be
/// Wrapped in an OMP Parallel section.
/// The large difference comes from the conditional checking the Cut_Off
/// Where it will change how tasks are spawned optimally
/// the array that is being modified \param arr
/// the starting position to sort  \param left
/// the final position to sort \param right
void QuickSort_Parallel(int* arr, int left, int right)
{
    int i = left;
    int j = right;
    int tmp;

    // A pivot element used to split the array into two parts
    const int pivot = arr[(left + right) / 2];

    // Partitions and swaps the elements from the ends down to the center
    while (i <= j)
    {
        while (arr[i] < pivot)
        {
            i++;
        }
        while (arr[j] > pivot)
        {
            j--;
        }
        if (i <= j)
        {
            tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            i++;
            j--;
        }
    }

    // Calls Quick sort like normal
    // until the cutoff has been met
    if ((right - left) < Cut_Off)
    {
        if (left < j)
        {
            QuickSort_Parallel(arr, left, j);
        }
        if (i < right)
        {
            QuickSort_Parallel(arr, i, right);
        }
    }
    else
    {
        // Once the array being sorted has reached the "cutoff"
        // it is then faster to handle the spawning of new tasks manually
        // instead of letting OpenMP do it automatically
        // "omp task" tells a single thread to handle the creation of new
        // tasks while the other threads perform the tasks
#pragma omp task
        {
            QuickSort_Parallel(arr, left, j);
        }
#pragma omp task
        {
            QuickSort_Parallel(arr, i, right);
        }
    }
}

// Calls Parallel version of QuickSort
//      and encloses it with OpenMP pragmas
void sort_p(int* arr)
{
    const int leftStart = 0;
    const int rightStart = Num_To_Sort - 1;

    /// Tells OpenMP to attempt to parallelize the enclosed code
    ///  using the specified max number of threads
#pragma omp parallel num_threads(omp_get_max_threads())
    {
        /// single: Causes the block of code to execute on one
        ///      thread only (Prevents nested loops/slowing performance)
        /// nowait: Modified the "single" omp call to not block more threads
        ///      from being used within the loop (Used in called QuickSort_Parallel)
#pragma omp single nowait
        {
            QuickSort_Parallel(arr, leftStart, rightStart);
        }
    }
}

int main() {
    int* arr_s = malloc(sizeof(int) * Num_To_Sort);
    const long chunk_size = Num_To_Sort / omp_get_max_threads();
#pragma omp parallel num_threads(omp_get_max_threads())
    {
        int p = omp_get_thread_num();
        unsigned int seed = (unsigned int)time(NULL) + (unsigned int)p;
        const long chunk_start = p * chunk_size;
        const long chunk_end = chunk_start + chunk_size;
        for (long i = chunk_start; i < chunk_end; i++) {
            arr_s[i] = rand_r(&seed);
        }
    }

    // Copy the array so that the sorting function can operate on it directly.
    // Note that this doubles the memory usage.
    // You may wish to test with slightly smaller arrays if you're running out of memory.
    int* arr_p = malloc(sizeof(int) * Num_To_Sort);
    memcpy(arr_p, arr_s, sizeof(int) * Num_To_Sort);

    struct timeval start, end;

    printf("Timing sequential...\n");
    gettimeofday(&start, NULL);
    sort_s(arr_s);
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double)(end.tv_usec - start.tv_usec) / 1000000);

    free(arr_s);

    printf("Timing parallel...\n");
    gettimeofday(&start, NULL);
    sort_p(arr_p);
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double)(end.tv_usec - start.tv_usec) / 1000000);

    free(arr_p);

    return 0;
}

