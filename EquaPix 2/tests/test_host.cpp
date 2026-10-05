#include "cpu_reference.h"
#include "metrics.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
static void require(bool ok, const char* name) { if (!ok) throw std::runtime_error(name); }
template<class F> void rejects(F f, const char* name) { bool caught=false; try{f();}catch(const std::exception&){caught=true;} require(caught,name); }
int main() { try {
 Image a{11,9,{}}; for(int y=0;y<a.height;++y) for(int x=0;x<a.width;++x) a.pixels.push_back(float(10+3*x+5*y));
 auto mask=makeMask(a, {"block",20,5,42});
 auto r=reconstructCPU(a,mask,1200);
 require(maxAbsoluteDifference(a,r.reconstructed)<0.002,"affine harmonic field recovery");
 for(std::size_t i=0;i<mask.size();++i) if(!mask[i]) require(a.pixels[i]==r.reconstructed.pixels[i],"known pixels fixed");
 auto m1=makeMask(a,{"random",30,5,73}),m2=makeMask(a,{"random",30,5,73});
 require(m1==m2,"deterministic masking");
 require(std::count(m1.begin(),m1.end(),1)==19,"exact missing count rounded over interior");
 for(int y=0;y<a.height;++y) for(int x=0;x<a.width;++x) if(x==0||y==0||x==a.width-1||y==a.height-1) require(!m1[y*a.width+x],"observed perimeter");
 auto unchanged=reconstructCPU(a,makeMask(a,{"random",0,5,1}),3);
 require(maxAbsoluteDifference(a,unchanged.reconstructed)==0,"zero missing identity");
 require(std::isinf(compareImages(a,a).psnr),"perfect PSNR infinity");
 Image b=a; b.pixels[0]+=10; require(std::abs(compareImages(a,b).rmse-10/std::sqrt(99.))<1e-8,"RMSE denominator");
 std::filesystem::create_directories("build/test_tmp");
 Image binary{3,2,{0,10,13,32,35,255}}; savePGM("build/test_tmp/p5.pgm",binary);
 require(loadPGM("build/test_tmp/p5.pgm").pixels==binary.pixels,"P5 whitespace-valued first pixels retained");
 {std::ofstream f("build/test_tmp/cr.pgm",std::ios::binary); f<<"P5\n3 1\n255\r"; f.put(char(10));f.put(char(30));f.put(char(40));}
 require(loadPGM("build/test_tmp/cr.pgm").pixels==std::vector<float>({10,30,40}),"P5 single CR delimiter retains initial LF-valued pixel");
 {std::ofstream f("build/test_tmp/crlf.pgm",std::ios::binary); f<<"P5\r\n3 1\r\n255\r\n";f.put(char(10));f.put(char(30));f.put(char(40));}
 require(loadPGM("build/test_tmp/crlf.pgm").pixels==std::vector<float>({10,30,40}),"P5 CRLF header");
 {std::ofstream f("build/test_tmp/p2.pgm"); f<<"P2\n# comment\n2 2\n15\n0 5 10 15\n";}
 require(loadPGM("build/test_tmp/p2.pgm").pixels==std::vector<float>({0,85,170,255}),"P2 scaling");
 {std::ofstream f("build/test_tmp/short.pgm"); f<<"P5\n3 3\n255\nX";}
 rejects([]{loadPGM("build/test_tmp/short.pgm");},"truncated PGM rejected");
 rejects([&]{makeMask(a,{"random",101,5,1});},"invalid percent rejected");
 rejects([&]{reconstructCPU(a,mask,0);},"invalid iterations rejected");
 Image tiny{1,1,{75}}; auto t=reconstructCPU(tiny,makeMask(tiny,{"block",20,64,1}),2);
 require(t.reconstructed.pixels[0]==75,"tiny image handling");
 saveMatrixPortrait("build/test_tmp/portrait.png",a,mask);
 // Independent PNG decoder verifies portrait dimensions in test_png_output.py.
 std::cout<<"PASS: host solver, masks, metrics, PGM, boundary and portrait checks\n";
 return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;} }
