#include "metrics.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
static void sameSize(const Image& a,const Image& b) { if(a.width!=b.width||a.height!=b.height||a.pixels.size()!=b.pixels.size()||a.pixels.empty()) throw std::runtime_error("Metric image dimensions mismatch"); }
Quality qualityFromMSE(double mse) {
 if(!std::isfinite(mse)||mse<0) throw std::runtime_error("Invalid MSE");
 return {std::sqrt(mse),mse==0?std::numeric_limits<double>::infinity():10*std::log10(255.0*255/mse)};
}
Quality compareImages(const Image& a,const Image& b) {
 sameSize(a,b);double sum=0;
 for(std::size_t i=0;i<a.pixels.size();++i){double d=double(a.pixels[i])-b.pixels[i];sum+=d*d;}
 return qualityFromMSE(sum/a.pixels.size());
}
double maxAbsoluteDifference(const Image& a,const Image& b) {
 sameSize(a,b);double m=0;for(std::size_t i=0;i<a.pixels.size();++i) {
  double d=std::abs(double(a.pixels[i])-b.pixels[i]);if(!std::isfinite(d)) throw std::runtime_error("Nonfinite reconstruction");m=std::max(m,d);
 }return m;
}
