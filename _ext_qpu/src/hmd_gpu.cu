#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "cuda_utils.h"

__global__ void hmd_kernel(int inSize, int memSize, int L,
                            const float *__restrict__ inputs,
                            float *__restrict__ mem){
    int inBias = (blockIdx.y * gridDim.x + blockIdx.x) * inSize * 4;
    int memBias = (blockIdx.y * gridDim.x + blockIdx.x) * memSize * 4;
    int idx = threadIdx.x * 2;
    int memIdx = idx + memSize / 2;
    int r = memBias + memIdx * 4;
    int inR = inBias + idx * 4;
    int tar;
    if (idx<inSize){
        mem[r]   = inputs[inR];
        mem[r+1]   = inputs[inR+1];
        mem[r+2]   = inputs[inR+2];
        mem[r+3]   = inputs[inR+3];
    }
    else{
        mem[r] = float(1);
    }
    if (idx+1<inSize){
        mem[r+4]   = inputs[inR+4];
        mem[r+5]   = inputs[inR+5];
        mem[r+6]   = inputs[inR+6];
        mem[r+7]   = inputs[inR+7];
    }
    else{
        mem[r+4] = float(1);
    }

    for(; L>0; L--){
        if ((memIdx & 1)==0){
            tar = memBias + memIdx*2;
            mem[tar]   = mem[r] * mem[r+4] - mem[r+1] * mem[r+5] - mem[r+2] * mem[r+6] - mem[r+3] * mem[r+7];
            mem[tar+1] = mem[r] * mem[r+5] + mem[r+1] * mem[r+4] + mem[r+2] * mem[r+7] - mem[r+3] * mem[r+6];
            mem[tar+2] = mem[r] * mem[r+6] - mem[r+1] * mem[r+7] + mem[r+2] * mem[r+4] + mem[r+3] * mem[r+5];
            mem[tar+3] = mem[r] * mem[r+7] + mem[r+1] * mem[r+6] - mem[r+2] * mem[r+5] + mem[r+3] * mem[r+4];
            memIdx /= 2;
            r = memBias + memIdx * 4;
        }
        __syncthreads();
    }

}

void hmd_kernel_wrapper(int B, int outSize, int inSize, int L, const float *inputs, float *mem){
    int memSize = 1<<(L+1);
    dim3 blockShape, threadShape;
    blockShape.y = B;
    blockShape.x = outSize;
    threadShape.x = memSize/4;

    hmd_kernel<<<blockShape,threadShape,0,at::cuda::getCurrentCUDAStream()>>>(inSize,memSize,L,inputs,mem);
}


// __global__ void hmd_grad_kernel_shared(int inSize, int memSize, int L, const float *__restrict__ grads,
//                                         const float *__restrict__ mem, float *__restrict__ lower){
//     int inBias = (blockIdx.y * gridDim.x + blockIdx.x) * inSize * 4;
//     int memBias = (blockIdx.y * gridDim.x + blockIdx.x) * memSize * 4;
//     int ptr = (threadIdx.x + memSize/2)^1;
//     int block = threadIdx.y;
//     int Lflag = ptr&1;
//     float mat[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
// }

// __global__ void hmd2quaternion(float *__restrict__ ou,const float *__restrict__ mem,float *__restrict__ in):


__global__ void hmd_grad_kernel(int inSize, int memSize, int L, const float *__restrict__ grads,
                                const float *__restrict__ mem, float *__restrict__ lower){
    int gradBias = (blockIdx.y * gridDim.x + blockIdx.x);
    int lowBias = (gradBias * inSize + threadIdx.x) << 2;
    int memBias = (gradBias * memSize) << 2;
    gradBias <<= 2;
    int ptr = (threadIdx.x + memSize/2)^1;
    int Rflag = ptr&1;
    float A[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float B[16] = {0};
    float *in = A, *ou = B, *temp=NULL;
    int posR = memBias + ptr * 4;
    for(;L>0;L--){
        if(Rflag){
            // Right(mem)*out
            ou[0] = mem[posR]*in[0]-mem[posR+1]*in[4]-mem[posR+2]*in[8] -mem[posR+3]*in[12];
            ou[1] = mem[posR]*in[1]-mem[posR+1]*in[5]-mem[posR+2]*in[9] -mem[posR+3]*in[13];
            ou[2] = mem[posR]*in[2]-mem[posR+1]*in[6]-mem[posR+2]*in[10]-mem[posR+3]*in[14];
            ou[3] = mem[posR]*in[3]-mem[posR+1]*in[7]-mem[posR+2]*in[11]-mem[posR+3]*in[15];

            ou[4] = mem[posR+1]*in[0]+mem[posR]*in[4]+mem[posR+3]*in[8] -mem[posR+2]*in[12];
            ou[5] = mem[posR+1]*in[1]+mem[posR]*in[5]+mem[posR+3]*in[9] -mem[posR+2]*in[13];
            ou[6] = mem[posR+1]*in[2]+mem[posR]*in[6]+mem[posR+3]*in[10]-mem[posR+2]*in[14];
            ou[7] = mem[posR+1]*in[3]+mem[posR]*in[7]+mem[posR+3]*in[11]-mem[posR+2]*in[15];

            ou[8]  = mem[posR+2]*in[0]-mem[posR+3]*in[4]+mem[posR]*in[8] +mem[posR+1]*in[12];
            ou[9]  = mem[posR+2]*in[1]-mem[posR+3]*in[5]+mem[posR]*in[9] +mem[posR+1]*in[13];
            ou[10] = mem[posR+2]*in[2]-mem[posR+3]*in[6]+mem[posR]*in[10]+mem[posR+1]*in[14];
            ou[11] = mem[posR+2]*in[3]-mem[posR+3]*in[7]+mem[posR]*in[11]+mem[posR+1]*in[15];

            ou[12] = mem[posR+3]*in[0]+mem[posR+2]*in[4]-mem[posR+1]*in[8] +mem[posR]*in[12];
            ou[13] = mem[posR+3]*in[1]+mem[posR+2]*in[5]-mem[posR+1]*in[9] +mem[posR]*in[13];
            ou[14] = mem[posR+3]*in[2]+mem[posR+2]*in[6]-mem[posR+1]*in[10]+mem[posR]*in[14];
            ou[15] = mem[posR+3]*in[3]+mem[posR+2]*in[7]-mem[posR+1]*in[11]+mem[posR]*in[15];
        }
        else{
            // Left(mem)*out
            ou[0] = mem[posR]*in[0]-mem[posR+1]*in[4]-mem[posR+2]*in[8] -mem[posR+3]*in[12];
            ou[1] = mem[posR]*in[1]-mem[posR+1]*in[5]-mem[posR+2]*in[9] -mem[posR+3]*in[13];
            ou[2] = mem[posR]*in[2]-mem[posR+1]*in[6]-mem[posR+2]*in[10]-mem[posR+3]*in[14];
            ou[3] = mem[posR]*in[3]-mem[posR+1]*in[7]-mem[posR+2]*in[11]-mem[posR+3]*in[15];

            ou[4] = mem[posR+1]*in[0]+mem[posR]*in[4]-mem[posR+3]*in[8] +mem[posR+2]*in[12];
            ou[5] = mem[posR+1]*in[1]+mem[posR]*in[5]-mem[posR+3]*in[9] +mem[posR+2]*in[13];
            ou[6] = mem[posR+1]*in[2]+mem[posR]*in[6]-mem[posR+3]*in[10]+mem[posR+2]*in[14];
            ou[7] = mem[posR+1]*in[3]+mem[posR]*in[7]-mem[posR+3]*in[11]+mem[posR+2]*in[15];

            ou[8]  = mem[posR+2]*in[0]+mem[posR+3]*in[4]+mem[posR]*in[8] -mem[posR+1]*in[12];
            ou[9]  = mem[posR+2]*in[1]+mem[posR+3]*in[5]+mem[posR]*in[9] -mem[posR+1]*in[13];
            ou[10] = mem[posR+2]*in[2]+mem[posR+3]*in[6]+mem[posR]*in[10]-mem[posR+1]*in[14];
            ou[11] = mem[posR+2]*in[3]+mem[posR+3]*in[7]+mem[posR]*in[11]-mem[posR+1]*in[15];

            ou[12] = mem[posR+3]*in[0]-mem[posR+2]*in[4]+mem[posR+1]*in[8] +mem[posR]*in[12];
            ou[13] = mem[posR+3]*in[1]-mem[posR+2]*in[5]+mem[posR+1]*in[9] +mem[posR]*in[13];
            ou[14] = mem[posR+3]*in[2]-mem[posR+2]*in[6]+mem[posR+1]*in[10]+mem[posR]*in[14];
            ou[15] = mem[posR+3]*in[3]-mem[posR+2]*in[7]+mem[posR+1]*in[11]+mem[posR]*in[15];
        }
        ptr = (ptr>>1)^1;
        posR = memBias + ptr * 4;
        Rflag = ptr&1;
        temp = ou;
        ou = in;
        in = temp;
    }
    lower[lowBias]   = in[0]*grads[gradBias]+in[1]*grads[gradBias+1]+in[2]*grads[gradBias+2]+in[3]*grads[gradBias+3];
    lower[lowBias+1] = in[4]*grads[gradBias]+in[5]*grads[gradBias+1]+in[6]*grads[gradBias+2]+in[7]*grads[gradBias+3];
    lower[lowBias+2] = in[8]*grads[gradBias]+in[9]*grads[gradBias+1]+in[10]*grads[gradBias+2]+in[11]*grads[gradBias+3];
    lower[lowBias+3] = in[12]*grads[gradBias]+in[13]*grads[gradBias+1]+in[14]*grads[gradBias+2]+in[15]*grads[gradBias+3];
}


void hmd_grad_kernel_wrapper(int B, int outSize, int inSize, int L, const float *grads, 
                                        const float *mem, float* lower){
    dim3 blockShape, threadShape;
    blockShape.y = B;
    blockShape.x = outSize;
    threadShape.x = inSize;

    hmd_grad_kernel<<<blockShape,threadShape,0,at::cuda::getCurrentCUDAStream()>>>(inSize,1<<(L+1),L,grads,mem,lower);
    
    // threadShape.y = 4;
    // hmd_grad_kernel_shared<<<blockShape,threadShape,0,at::cuda::getCurrentCUDAStream()>>>();
}