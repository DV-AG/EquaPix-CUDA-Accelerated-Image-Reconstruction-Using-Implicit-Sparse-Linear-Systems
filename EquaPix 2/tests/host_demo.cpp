// Produces explicitly CPU-only reference illustrations, never CUDA proof.
#include "cpu_reference.h"
#include "metrics.h"
#include <filesystem>
#include <iostream>
#include <fstream>
int main(){namespace fs=std::filesystem;try{
 fs::path out="results/host_reference_examples";fs::create_directories(out);
 std::ofstream table(out/"cpu_reference_metrics.csv");table<<"image_name,backend,iterations,cpu_ms,rmse,psnr\n";
 for(const auto& name:{"image_0000_gradient.pgm","image_0007_smooth.pgm"}){
  auto im=loadPGM((fs::path("data/sample")/name).string());auto mask=makeMask(im,{"block",20,32,42});auto r=reconstructCPU(im,mask,1500);auto q=compareImages(im,r.reconstructed);
  savePNG((out/resultImageName(name,"original")).string(),im);savePNG((out/resultImageName(name,"masked")).string(),maskedImage(im,mask));savePNG((out/resultImageName(name,"cpu_reconstructed")).string(),r.reconstructed);
  saveMatrixPortrait((out/resultImageName(name,"matrix")).string(),im,mask);
  table<<name<<",CPU_REFERENCE_ONLY,1500,"<<r.solver_ms<<','<<q.rmse<<','<<q.psnr<<'\n';
  std::cout<<"CPU REFERENCE ONLY: "<<name<<" | iterations=1500 | CPU ms="<<r.solver_ms<<" | RMSE="<<q.rmse<<" | PSNR="<<q.psnr<<'\n';
 }
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
