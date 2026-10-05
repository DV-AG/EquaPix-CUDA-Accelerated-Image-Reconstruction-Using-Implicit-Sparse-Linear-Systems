#include "cpu_reference.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <stdexcept>
static void validateMask(const Image& image,const std::vector<unsigned char>& mask) {
 validateImage(image); if(mask.size()!=image.pixels.size()) throw std::runtime_error("Mask size mismatch");
 for(int y=0;y<image.height;++y) for(int x=0;x<image.width;++x) {
  auto m=mask[static_cast<std::size_t>(y)*image.width+x];
  if(m>1||(m&&(x==0||y==0||x==image.width-1||y==image.height-1))) throw std::runtime_error("Mask must be binary with observed perimeter");
 }
}
std::vector<unsigned char> makeMask(const Image& image,const MaskConfig& config) {
 validateImage(image);
 if(!std::isfinite(config.missing_percent)||config.missing_percent<0||config.missing_percent>100||config.block_size<=0) throw std::runtime_error("Invalid mask parameters");
 std::vector<unsigned char> mask(image.pixels.size(),0);
 if(config.type=="random") {
  std::vector<std::size_t> indices;
  for(int y=1;y<image.height-1;++y) for(int x=1;x<image.width-1;++x) indices.push_back(static_cast<std::size_t>(y)*image.width+x);
  // Explicit Fisher-Yates with rejection sampling: stable across C++ libraries.
  std::mt19937 rng(config.seed);
  for(std::size_t i=indices.size();i>1;--i) {
   auto bound=static_cast<std::uint32_t>(i); auto threshold=static_cast<std::uint32_t>(-bound)%bound; std::uint32_t r;
   do{r=rng();}while(r<threshold); std::swap(indices[i-1],indices[r%bound]);
  }
  auto count=static_cast<std::size_t>(std::llround(indices.size()*config.missing_percent/100.0));
  for(std::size_t i=0;i<count;++i) mask[indices[i]]=1;
 } else if(config.type=="block") {
  int bw=std::min(config.block_size,std::max(0,image.width-2));
  int bh=std::min(config.block_size,std::max(0,image.height-2));
  int x0=(image.width-bw)/2,y0=(image.height-bh)/2;
  for(int y=y0;y<y0+bh;++y) for(int x=x0;x<x0+bw;++x) mask[static_cast<std::size_t>(y)*image.width+x]=1;
 } else throw std::runtime_error("Mask must be random or block");
 return mask;
}
float observedMean(const Image& image,const std::vector<unsigned char>& mask) {
 validateMask(image,mask); double sum=0;std::size_t count=0;
 for(std::size_t i=0;i<mask.size();++i) if(!mask[i]){sum+=image.pixels[i];++count;}
 return count?static_cast<float>(sum/count):0.f;
}
Image maskedImage(const Image& image,const std::vector<unsigned char>& mask) {
 validateMask(image,mask);Image out=image;for(std::size_t i=0;i<mask.size();++i) if(mask[i])out.pixels[i]=0;return out;
}
CpuResult reconstructCPU(const Image& image,const std::vector<unsigned char>& mask,int iterations) {
 if(iterations<=0) throw std::runtime_error("Iterations must be positive");
 float mean=observedMean(image,mask);Image current=image,next=image;
 for(std::size_t i=0;i<mask.size();++i) if(mask[i]) current.pixels[i]=next.pixels[i]=mean;
 auto start=std::chrono::steady_clock::now();
 for(int k=0;k<iterations;++k) {
  for(int y=1;y<image.height-1;++y) for(int x=1;x<image.width-1;++x) {
   auto i=static_cast<std::size_t>(y)*image.width+x;
   if(mask[i]) next.pixels[i]=0.25f*(current.pixels[i-1]+current.pixels[i+1]+current.pixels[i-image.width]+current.pixels[i+image.width]);
  }
  current.pixels.swap(next.pixels);
 }
 auto stop=std::chrono::steady_clock::now();
 return {std::move(current),std::chrono::duration<double,std::milli>(stop-start).count()};
}
void saveMatrixPortrait(const std::string& path,const Image& image,const std::vector<unsigned char>& mask) {
 validateMask(image,mask);
 constexpr int side=12,n=side*side;Image portrait{n,n,std::vector<float>(n*n,0)};
 // Raster of a representative 12x12 system. Only write its nonzero locations;
 // the production NxN matrix is never constructed.
 for(int y=0;y<side;++y) for(int x=0;x<side;++x) {
  int row=y*side+x;portrait.pixels[row*n+row]=255;
  int sx=x*(image.width-1)/(side-1),sy=y*(image.height-1)/(side-1);
  if(x>0&&y>0&&x<side-1&&y<side-1&&mask[static_cast<std::size_t>(sy)*image.width+sx]) {
   for(int col:{row-1,row+1,row-side,row+side}) portrait.pixels[row*n+col]=140;
  }
 }
 savePNG(path,portrait);
}
