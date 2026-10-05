#pragma once
#include "image_io.h"
#include <cstdint>
struct MaskConfig { std::string type="random"; double missing_percent=20; int block_size=64; std::uint32_t seed=42; };
std::vector<unsigned char> makeMask(const Image& image, const MaskConfig& config);
float observedMean(const Image& image, const std::vector<unsigned char>& mask);
Image maskedImage(const Image& image, const std::vector<unsigned char>& mask);
struct CpuResult { Image reconstructed; double solver_ms=0; };
CpuResult reconstructCPU(const Image& image, const std::vector<unsigned char>& mask, int iterations);
void saveMatrixPortrait(const std::string& path, const Image& image, const std::vector<unsigned char>& mask);
