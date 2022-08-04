#ifndef SIFT_CUDA_H
#define SIFT_CUDA_H

#include "DataTypes.cuh"

__global__ void MakeKeypointKernel(siftPar* param, keypointHolder* holder, keypoint* key, float* grad, float* ori);

__device__ void MakeKeypointSampleKernel(
	keypoint& key, const float* grad, const float* ori,
	int nwidth, int nheight, float scale, float row, float
	col, siftPar& par);

__device__ void KeySampleVecKernel(keypoint& key, const float* grad, const float* ori, int nwidth, int nheight, float scale, float row, float col, siftPar& par);

__device__ void KeySampleKernel(keypoint& key, const float* grad, const float* ori, int nwidth, int nheight, float scale, float row, float col, siftPar& par);

__device__ void AddSampleKernel(keypoint& key, const float* grad, const float* orim, int nwidth, int nheight, int r, int c, float rpos, float cpos, float rx, float cx, siftPar& par);

__device__ void PlaceInIndexKernel(float *index, float mag, float ori, float rx, float cx, siftPar& par);

__device__ void NormalizeVecKernel(float* vec);

#endif