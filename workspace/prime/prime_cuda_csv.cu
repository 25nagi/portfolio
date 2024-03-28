#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <windows.h> // Windows APIの時間関数にアクセスするために必要

#define Grid_x 1024
#define Grid_y 1024
#define Block_x 16
#define Block_y 8

__global__ void thread_num(unsigned long long int *device_result, unsigned long long int cycle, unsigned long long int thread_size);

/* timer */
int timer(void){
    time_t now = time(NULL);
    struct tm *pnow = localtime(&now);
    char buff[128] = "";
    sprintf(buff, "%d:%d:%d", pnow->tm_hour, pnow->tm_min, pnow->tm_sec);
    printf("%s\n", buff);
    return 0;
}

int main(int argc, char **argv){
    if (argc < 2) {
        printf("Usage: %s <number>\n", argv[0]);
        return 1;
    }

    unsigned long long int num = atoll(argv[1]);
    unsigned long long int thread_size = Grid_x * Grid_y * Block_x * Block_y;
    unsigned long long int cycle = num / thread_size + (num % thread_size > 0);
    unsigned long long int *host_result = (unsigned long long int *)malloc(thread_size * sizeof(unsigned long long int));
    unsigned long long int *device_result;
    checkCudaErrors(cudaMalloc((void **)&device_result, thread_size * sizeof(unsigned long long int)));

    FILE *csvfile = fopen("prime_times.csv", "w");
    if (csvfile == NULL) {
        printf("Cannot open csv file\n");
        return 1;
    }
    fprintf(csvfile, "Index,Prime,Time,OriginalTime\n");

    LARGE_INTEGER start_time, current_time, frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start_time);

    unsigned long long int primeIndex = 0;
    long long first_prime_time_ms = -1; // 最初の素数の時間を格納する変数を初期化
    long long elapsed_ms = 0; // 経過時間

    for (unsigned long long int i = 0; i < cycle; i++) {
        thread_num<<<Grid_x, dim3(Block_x, Block_y)>>>(device_result, i, thread_size);
        cudaDeviceSynchronize();

        checkCudaErrors(cudaMemcpy(host_result, device_result, thread_size * sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

        for (unsigned long long int j = 0; j < thread_size; j++) {
            if (host_result[j] != 0) {
                primeIndex++;
                QueryPerformanceCounter(&current_time);
                long long original_time_ms = ((current_time.QuadPart - start_time.QuadPart) * 1000) / frequency.QuadPart;
                // 最初の素数の時間が未設定なら設定する
                if (first_prime_time_ms == -1) {
                    first_prime_time_ms = original_time_ms;
                }
                elapsed_ms = original_time_ms - first_prime_time_ms;
                fprintf(csvfile, "%llu,%llu,%lld,%lld\n", primeIndex, host_result[j], elapsed_ms, original_time_ms);
            }
        }
    }

    fclose(csvfile);
    free(host_result);
    cudaFree(device_result);

    return 0;
}

__global__ void thread_num(unsigned long long int *device_result, unsigned long long int cycle, unsigned long long int thread_size){
    unsigned long long int thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned long long int thread_idy = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned long long int thread_idz = threadIdx.z + blockDim.z * blockIdx.z;
    unsigned long long int global_thread_id = thread_idx + Grid_x * Block_x * (thread_idy + Grid_y * Block_y * thread_idz);
    unsigned long long int num_to_check = global_thread_id + cycle * thread_size;
    int flag = 0;

    if (num_to_check < 2) {
        device_result[global_thread_id] = 0;
    } else if (num_to_check == 2) {
        device_result[global_thread_id] = 2;
    } else if (num_to_check % 2 == 0) {
        device_result[global_thread_id] = 0;
    } else {
        unsigned long long int dev = 3;
        while ((dev * dev) <= num_to_check) {
            if (num_to_check % dev == 0) {
                flag = 1;
                break;
            }
            dev += 2;
        }
        if (flag == 0) {
            device_result[global_thread_id] = num_to_check;
        } else {
            device_result[global_thread_id] = 0;
        }
    }
}
