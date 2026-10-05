#include "kernels.cuh"
#include "cpu_reference.h"
#include "metrics.h"
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <set>
#include <stdexcept>
namespace fs=std::filesystem;
struct Options { fs::path input="data/sample",output="results";MaskConfig mask;int iterations=500,cpu_count=10; };
static void usage(){std::cout<<
 "EquaPix: CUDA sparse image reconstruction\n"
 "Usage: ./equapix [--input DIR] [--output DIR] [--mask random|block]\n"
 "  [--missing-percent 0..100] [--block-size N] [--iterations N]\n"
 "  [--seed UINT32] [--cpu-benchmark-count N] [--help]\n"
 "Defaults: data/sample, results, random, 20%, block 64, 500 iterations,\n"
 "seed 42, 10 CPU comparisons. Perimeter always observed. No CPU fallback.\n";}
static int parseInt(const std::string& s){std::size_t pos;long long v=std::stoll(s,&pos);if(pos!=s.size()||v<0||v>std::numeric_limits<int>::max())throw std::runtime_error("Invalid nonnegative integer: "+s);return int(v);}
static Options parse(int argc,char** argv){Options o;
 for(int i=1;i<argc;++i){std::string key=argv[i];if(key=="--help"){usage();std::exit(0);}if(i+1==argc)throw std::runtime_error("Missing value for "+key);std::string v=argv[++i];
  if(key=="--input")o.input=v;else if(key=="--output")o.output=v;else if(key=="--mask")o.mask.type=v;
  else if(key=="--iterations")o.iterations=parseInt(v);else if(key=="--cpu-benchmark-count")o.cpu_count=parseInt(v);
  else if(key=="--block-size")o.mask.block_size=parseInt(v);
  else if(key=="--missing-percent"){std::size_t pos;o.mask.missing_percent=std::stod(v,&pos);if(pos!=v.size())throw std::runtime_error("Invalid missing percent");}
  else if(key=="--seed"){std::size_t pos;auto s=std::stoull(v,&pos);if(v.empty()||v[0]=='-'||pos!=v.size()||s>std::numeric_limits<std::uint32_t>::max())throw std::runtime_error("Seed outside UINT32 range");o.mask.seed=static_cast<std::uint32_t>(s);}
  else throw std::runtime_error("Unknown option: "+key);
 }
 if(o.iterations<1||o.mask.block_size<1||!std::isfinite(o.mask.missing_percent)||o.mask.missing_percent<0||o.mask.missing_percent>100||(o.mask.type!="random"&&o.mask.type!="block"))throw std::runtime_error("Invalid configuration; see --help");
 if(!fs::is_directory(o.input))throw std::runtime_error("Input directory does not exist: "+o.input.string());
 if(fs::weakly_canonical(o.input)==fs::weakly_canonical(o.output))throw std::runtime_error("Input and output directories must differ");
 return o;
}
static std::string csv(const std::string& s){std::string out="\"";for(char c:s){if(c=='\"')out+='\"';out+=c;}return out+'\"';}
static std::uint32_t imageSeed(std::uint32_t seed,const std::string& name){std::uint32_t hash=2166136261u;for(unsigned char c:name){hash^=c;hash*=16777619u;}return seed^hash;}
static void cudaCheck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(int argc,char** argv){std::ofstream log;
 auto report=[&](const std::string& s){std::cout<<s<<'\n';if(log)log<<s<<'\n';};
 try{auto o=parse(argc,argv);
  fs::create_directories(o.output/"examples");fs::create_directories(o.output/"matrix_portraits");
  log.open(o.output/"execution_log.txt");if(!log)throw std::runtime_error("Cannot create execution log");
  std::ostringstream command;command<<"Command:";for(int i=0;i<argc;++i)command<<' '<<argv[i];report(command.str());
  int count=0;cudaCheck(cudaGetDeviceCount(&count));if(!count)throw std::runtime_error("No CUDA GPU found; this executable requires CUDA");
  cudaCheck(cudaSetDevice(0));cudaDeviceProp prop{};cudaCheck(cudaGetDeviceProperties(&prop,0));cudaCheck(cudaFree(nullptr));
  report(std::string("CUDA device: ")+prop.name+" | compute capability "+std::to_string(prop.major)+"."+std::to_string(prop.minor));
  std::vector<fs::path> images;
  for(const auto& entry:fs::directory_iterator(o.input)){auto ext=entry.path().extension().string();for(char& c:ext)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));if(entry.is_regular_file()&&ext==".pgm")images.push_back(entry.path());}
  std::sort(images.begin(),images.end());if(images.empty())throw std::runtime_error("No .pgm images found");
  std::set<std::string> output_stems;
  for(const auto& path:images) if(!output_stems.insert(path.stem().string()).second) throw std::runtime_error("Duplicate input image stem would overwrite PNG outputs: "+path.stem().string());
  report("Images found: "+std::to_string(images.size()));
  report("Mask: "+o.mask.type+" | requested interior missing percent: "+std::to_string(o.mask.missing_percent)+" | block size: "+std::to_string(o.mask.block_size)+" | seed: "+std::to_string(o.mask.seed));
  report("Jacobi iterations: "+std::to_string(o.iterations)+" | CPU benchmark count: "+std::to_string(o.cpu_count));
  std::ofstream table(o.output/"metrics.csv");if(!table)throw std::runtime_error("Cannot create metrics.csv");
  table<<"image_name,mask_type,requested_missing_percent,actual_missing_percent,width,height,iterations,gpu_ms,gpu_pipeline_ms,total_ms,rmse,psnr,missing_rmse,missing_psnr,cpu_ms,speedup,cpu_gpu_max_abs,validation,unknowns,implicit_nonzeros,image_seed\n"<<std::setprecision(10);
  std::size_t done=0,failed=0,benchmarked=0;double sum_rmse=0,sum_psnr=0,gpu_total=0,cpu_total=0,gpu_compared=0;
  for(const auto& path:images){try{
   auto begin=std::chrono::steady_clock::now();auto im=loadPGM(path.string());auto cfg=o.mask;cfg.seed=imageSeed(o.mask.seed,path.filename().string());auto mask=makeMask(im,cfg);
   auto gpu=reconstructGPU(im,mask,o.iterations);auto quality=qualityFromMSE(gpu.mse);
   auto unknown=std::count(mask.begin(),mask.end(),1);double missing_sum=0;
   for(std::size_t i=0;i<mask.size();++i)if(mask[i]){double d=double(im.pixels[i])-gpu.reconstructed.pixels[i];missing_sum+=d*d;}
   auto missing_quality=qualityFromMSE(unknown?missing_sum/unknown:0);double cpu_ms=0,speedup=0,diff=0;bool compared=benchmarked<static_cast<std::size_t>(o.cpu_count);
   if(compared){auto cpu=reconstructCPU(im,mask,o.iterations);cpu_ms=cpu.solver_ms;diff=maxAbsoluteDifference(cpu.reconstructed,gpu.reconstructed);
    if(diff>0.002)throw std::runtime_error("CPU/GPU disagreement exceeds 0.002 intensity units: "+std::to_string(diff));
    speedup=gpu.solver_ms>0?cpu_ms/gpu.solver_ms:0;++benchmarked;cpu_total+=cpu_ms;gpu_compared+=gpu.solver_ms;
   }
   std::string name=path.filename().string();
   savePNG((o.output/"examples"/resultImageName(name,"original")).string(),im);
   savePNG((o.output/"examples"/resultImageName(name,"masked")).string(),gpu.masked);
   savePNG((o.output/"examples"/resultImageName(name,"reconstructed")).string(),gpu.reconstructed);
   saveMatrixPortrait((o.output/"matrix_portraits"/resultImageName(name,"matrix")).string(),im,mask);
   double total=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
   table<<csv(name)<<','<<cfg.type<<',';if(cfg.type=="random")table<<cfg.missing_percent;
   table<<','<<100.0*unknown/im.pixels.size()<<','<<im.width<<','<<im.height<<','<<o.iterations<<','<<gpu.solver_ms<<','<<gpu.pipeline_ms<<','<<total<<','<<quality.rmse<<','<<quality.psnr<<','<<missing_quality.rmse<<','<<missing_quality.psnr<<',';
   if(compared)table<<cpu_ms<<','<<speedup<<','<<diff<<",PASS";else table<<",,,NOT_RUN";
   table<<','<<unknown<<','<<im.pixels.size()+4*static_cast<std::size_t>(unknown)<<','<<cfg.seed<<'\n';if(!table)throw std::runtime_error("Metrics write failed");
   std::ostringstream line;line<<std::fixed<<std::setprecision(4)<<"["<<done+1<<"/"<<images.size()<<"] "<<name<<" "<<im.width<<"x"<<im.height<<" | unknowns="<<unknown<<" | GPU solve="<<gpu.solver_ms<<" ms | pipeline="<<gpu.pipeline_ms<<" ms | RMSE="<<quality.rmse<<" | PSNR="<<quality.psnr<<" dB";
   if(compared) { line<<" | CPU="<<cpu_ms<<" ms | speedup="<<speedup<<"x | max_abs="<<diff<<" PASS"; }
   report(line.str());
   ++done;sum_rmse+=quality.rmse;sum_psnr+=quality.psnr;gpu_total+=gpu.solver_ms;
  }catch(const std::exception& e){++failed;report("FAILED "+path.filename().string()+": "+e.what());}}
  report("Processed successfully: "+std::to_string(done)+" | failed: "+std::to_string(failed));
  if(done){report("Average RMSE: "+std::to_string(sum_rmse/done)+" | average PSNR: "+std::to_string(sum_psnr/done)+" dB | summed GPU solver time: "+std::to_string(gpu_total)+" ms");}
  if(benchmarked&&gpu_compared>0)report("Compared images: "+std::to_string(benchmarked)+" | aggregate CPU/GPU solver speedup: "+std::to_string(cpu_total/gpu_compared)+"x");
  report("Outputs: "+fs::absolute(o.output).string()+" (metrics.csv, execution_log.txt, examples/, matrix_portraits/)");
  return failed||!done?2:0;
 }catch(const std::exception& e){report(std::string("ERROR: ")+e.what());return 1;}}
