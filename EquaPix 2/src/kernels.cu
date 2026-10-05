#include "kernels.cuh"
#include "cpu_reference.h"
#include <cuda_runtime.h>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
static void check(cudaError_t code,const char* what) {
 if(code!=cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(code));
}
#define CUDA_CHECK(call) check((call),#call)
template<class T> class DeviceBuffer {
 T* ptr_=nullptr;
 public: explicit DeviceBuffer(std::size_t count) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_),count*sizeof(T))); }
 ~DeviceBuffer(){ if(ptr_) cudaFree(ptr_); }
 DeviceBuffer(const DeviceBuffer&)=delete;DeviceBuffer& operator=(const DeviceBuffer&)=delete;
 T* get() const {return ptr_;}
};
class Event {
 cudaEvent_t event_{};
 public: Event(){CUDA_CHECK(cudaEventCreate(&event_));}~Event(){cudaEventDestroy(event_);}
 Event(const Event&)=delete;Event& operator=(const Event&)=delete;
 operator cudaEvent_t() const{return event_;}
};
__global__ void applyMaskKernel(const float* original,const unsigned char* mask,float* masked,int n) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<static_cast<unsigned int>(n)) masked[i]=mask[i]?0.f:original[i];
}
__global__ void initializeUnknownPixelsKernel(const float* masked,const unsigned char* mask,float* current,float* next,float mean,int n) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<static_cast<unsigned int>(n)) current[i]=next[i]=mask[i]?mean:masked[i];
}
__global__ void jacobiIterationKernel(const float* current,float* next,const unsigned char* mask,int w,int h,int n) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=static_cast<unsigned int>(n))return;
 int x=i%w,y=i/w;
 // Explicit border guard even though masks always preserve the perimeter.
 if(!mask[i]||x==0||y==0||x==w-1||y==h-1){next[i]=current[i];return;}
 next[i]=0.25f*(current[i-1]+current[i+1]+current[i-w]+current[i+w]);
}
__global__ void computeSquaredErrorKernel(const float* original,const float* result,double* block_sums,int n) {
 __shared__ double partial[256];
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;unsigned int t=threadIdx.x;
 double d=i<static_cast<unsigned int>(n)?double(original[i])-double(result[i]):0.0;partial[t]=d*d;
 __syncthreads();
 // One block sum per 256 pixels avoids a large error vector and float atomics.
 for(unsigned int stride=blockDim.x/2;stride>0;stride/=2){if(t<stride)partial[t]+=partial[t+stride];__syncthreads();}
 if(t==0)block_sums[blockIdx.x]=partial[0];
}
GpuResult reconstructGPU(const Image& image,const std::vector<unsigned char>& mask,int iterations) {
 if(iterations<=0)throw std::runtime_error("Iterations must be positive");
 float mean=observedMean(image,mask); // validates image and anchored binary mask
 auto begin=std::chrono::steady_clock::now();
 int n=static_cast<int>(image.pixels.size()),blocks=(n-1)/256+1;
 DeviceBuffer<float> original(n),masked(n),a(n),b(n);DeviceBuffer<unsigned char> dm(n);DeviceBuffer<double> sums(blocks);
 CUDA_CHECK(cudaMemcpy(original.get(),image.pixels.data(),n*sizeof(float),cudaMemcpyHostToDevice));
 CUDA_CHECK(cudaMemcpy(dm.get(),mask.data(),n*sizeof(unsigned char),cudaMemcpyHostToDevice));
 applyMaskKernel<<<blocks,256>>>(original.get(),dm.get(),masked.get(),n);CUDA_CHECK(cudaGetLastError());
 initializeUnknownPixelsKernel<<<blocks,256>>>(masked.get(),dm.get(),a.get(),b.get(),mean,n);CUDA_CHECK(cudaGetLastError());
 Event start,stop;float* current=a.get();float* next=b.get();
 CUDA_CHECK(cudaEventRecord(start));
 for(int k=0;k<iterations;++k){
  jacobiIterationKernel<<<blocks,256>>>(current,next,dm.get(),image.width,image.height,n);
  CUDA_CHECK(cudaGetLastError());std::swap(current,next);
 }
 CUDA_CHECK(cudaEventRecord(stop));CUDA_CHECK(cudaEventSynchronize(stop));float ms;
 CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
 computeSquaredErrorKernel<<<blocks,256>>>(original.get(),current,sums.get(),n);CUDA_CHECK(cudaGetLastError());
 std::vector<double> host_sums(blocks);CUDA_CHECK(cudaMemcpy(host_sums.data(),sums.get(),blocks*sizeof(double),cudaMemcpyDeviceToHost));
 GpuResult result;result.reconstructed=image;result.masked=image;
 CUDA_CHECK(cudaMemcpy(result.reconstructed.pixels.data(),current,n*sizeof(float),cudaMemcpyDeviceToHost));
 CUDA_CHECK(cudaMemcpy(result.masked.pixels.data(),masked.get(),n*sizeof(float),cudaMemcpyDeviceToHost));
 double sum=0;for(double s:host_sums)sum+=s;result.mse=sum/n;result.solver_ms=ms;
 result.pipeline_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
 return result;
}
