#pragma once
#include "image_io.h"
struct GpuResult { Image reconstructed, masked; double solver_ms=0, pipeline_ms=0, mse=0; };
GpuResult reconstructGPU(const Image& image,const std::vector<unsigned char>& mask,int iterations);
