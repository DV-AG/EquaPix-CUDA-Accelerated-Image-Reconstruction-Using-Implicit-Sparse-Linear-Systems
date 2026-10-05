#pragma once
#include "image_io.h"
struct Quality { double rmse=0, psnr=0; };
Quality qualityFromMSE(double mse);
Quality compareImages(const Image& reference, const Image& result);
double maxAbsoluteDifference(const Image& a, const Image& b);
