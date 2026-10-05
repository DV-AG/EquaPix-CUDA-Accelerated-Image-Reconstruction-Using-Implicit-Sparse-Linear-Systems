#include "image_io.h"
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
static void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
template<class F> void rejects(F f,const char* msg){bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,msg);}
int main(){try{
 std::filesystem::create_directories("build/png_test");
 require(resultImageName("image_001.pgm","original")=="image_001_original.png","clean PNG filename");
 Image small{4,2,{-5,0,10,13,32,35,127.5f,300}};savePNG("build/png_test/small.png",small);
 Image large{256,256,std::vector<float>(256*256)};
 for(std::size_t i=0;i<large.pixels.size();++i)large.pixels[i]=float(i%256);
 savePNG("build/png_test/multiblock.png",large);
 savePNG("build/png_test/tiny.png",{1,1,{75}});
 rejects([]{savePNG("build/png_test/bad.png",{2,2,{1}});},"reject incomplete image");
 rejects([]{savePNG("build/png_test/bad.png",{1,1,{std::numeric_limits<float>::quiet_NaN()}});},"reject nonfinite pixel");
 rejects([]{savePNG("build/nonexistent_parent/bad.png",{1,1,{75}});},"report failed writes");
 std::cout<<"PASS: native PNG writer fixtures, naming, and error handling\n";
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
