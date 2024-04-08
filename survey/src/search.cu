#include <stdio.h>
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



// Read in a file
// Split it into sections
// 

