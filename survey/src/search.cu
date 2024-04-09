#include <stdio.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include "time.h"
#include "search.cuh"

void generate_prefix_table(char * pattern, int len, int prefix_tbl[]){
    prefix_tbl[0] = -1;
    int i = 0;
    int j = -1;

    while(i < len){
        if(j == -1 || pattern[i] == pattern[j]){
            i += 1;
            j += 1;
            prefix_tbl[i] = j;
        }else{
            j = prefix_tbl[j];
        }
    }
}

__global__ void KMP(char* pattern, char* target,int f[],int c[],int n, int m)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int i = n * index;
    int j = n * (index + 2)-1;
    if(i>m)
        return;
    if(j>m)
        j=m;
    int k = 0;        
    while (i < j)
    {
        if (k == -1)
        {
            i++;
            k = 0;
        }
        else if (target[i] == pattern[k])
        {
            i++;
            k++;
            if (k == n)
            {
                c[i - n] = i-n;
                i = i - k + 1;
            }
        }
        else
            k = f[k];
    }
    return;
}
 
int main(int argc, char* argv[])
{
    FILE *fptr;
    char *host_pattern;
    fptr = fopen("./rsrc/input.txt", "r"); 
    if(argc < 2){
        return EXIT_FAILURE;
    }
    host_pattern = argc[1];
    if(fptr == NULL){
        fprintf(stderr, "Error reading from file address\n");
        return EXIT_FAILURE;
    }else{
        fprintf(stderr, "Success in opening file.\n");
    }
    int *host_prefix_tbl = (int *) malloc(sizeof(int) * strlen(host_pattern));
    generate_prefix_table(host_pattern, strlen(host_pattern), host_prefix_tbl);

    char *host_buffer = (char *) malloc(sizeof(char) * 128000);
    int ch;
    int curr_idx;
    do {
        ch = fgetc(fptr);
        printf("%c", ch);
        host_buffer[curr_idx] = ch;
        curr_idx++;
    } while (ch != EOF);

    char *dev_buffer;
    char *dev_pattern;
    char *dev_prefix_tbl;

    cudaMalloc(&dev_buffer, curr_idx); 
    cudaMemcpy(host_buffer, dev_buffer, curr_idx, cudaMemcpyHostToDevice);

    cudaMalloc(&dev_pattern, strlen(host_pattern));
    cudaMemcpy(host_pattern, dev_pattern, strlen(host_pattern), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_prefix_tbl, strlen(host_pattern));
    cudaMemcpy(host_prefix_tbl, dev_prefix_tbl, strlen(host_pattern), cudaMemcpyHostToDevice);
    KMP<<<256, 256>>>(dev_buffer, dev_pattern);

    cudaFree(dev_buffer);
    cudaFree(dev_pattern);
    cudaFree(dev_prefix_tbl);
    free(host_buffer);
    free(host_pattern);
    free(host_prefix_tbl);
    return EXIT_SUCCESS;
}

