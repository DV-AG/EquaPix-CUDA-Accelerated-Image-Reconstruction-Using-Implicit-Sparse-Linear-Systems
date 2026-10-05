#pragma once
#include <cstddef>
#include <string>
#include <vector>
struct Image { int width=0, height=0; std::vector<float> pixels; };
Image loadPGM(const std::string& path);
void savePGM(const std::string& path, const Image& image);
void validateImage(const Image& image);

// Native 8-bit grayscale PNG output; no external library or conversion step.
void savePNG(const std::string& path, const Image& image);
std::string resultImageName(const std::string& input_name, const std::string& kind);
