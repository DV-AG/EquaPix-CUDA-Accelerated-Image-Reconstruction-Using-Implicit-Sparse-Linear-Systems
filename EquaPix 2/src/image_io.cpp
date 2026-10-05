#include "image_io.h"
#include <fstream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cctype>
#include <algorithm>
static std::string token(std::istream& in, bool fold_crlf=true, char* delimiter=nullptr) {
 std::string s; char c;
 while(in.get(c)) { if(std::isspace(static_cast<unsigned char>(c))) continue;
  if(c=='#'){in.ignore(std::numeric_limits<std::streamsize>::max(),'\n');continue;} s+=c;break; }
 while(in.get(c)) { if(std::isspace(static_cast<unsigned char>(c))) { if(delimiter) *delimiter=c; if(fold_crlf&&c=='\r'&&in.peek()=='\n') in.get(); break; }
  if(c=='#'){in.ignore(std::numeric_limits<std::streamsize>::max(),'\n');break;} s+=c; }
 if(s.empty()) throw std::runtime_error("Incomplete PGM header or samples");
 return s;
}
static int integer(const std::string& s) { std::size_t used; int v=std::stoi(s,&used); if(used!=s.size()) throw std::runtime_error("Invalid integer in PGM"); return v; }
void validateImage(const Image& image) {
 if(image.width<=0||image.height<=0||static_cast<long long>(image.width)*image.height>std::numeric_limits<int>::max()) throw std::runtime_error("Invalid or too large image dimensions");
 if(image.pixels.size()!=static_cast<std::size_t>(image.width)*image.height) throw std::runtime_error("Image pixel count mismatch");
 for(float p:image.pixels) if(!std::isfinite(p)||p<0||p>255) throw std::runtime_error("Image samples must be finite and in [0,255]");
}
Image loadPGM(const std::string& path) {
 std::ifstream in(path,std::ios::binary); if(!in) throw std::runtime_error("Cannot open image: "+path);
 auto magic=token(in); if(magic!="P5"&&magic!="P2") throw std::runtime_error("Only P5/P2 PGM supported: "+path);
 Image im; im.width=integer(token(in)); im.height=integer(token(in)); char separator=0; int maxval=integer(token(in,magic!="P5",&separator));
 if(im.width<=0||im.height<=0||static_cast<long long>(im.width)*im.height>std::numeric_limits<int>::max()||maxval<=0||maxval>65535) throw std::runtime_error("Invalid PGM header: "+path);

 // A P5 header has one raster separator. For common CRLF files, fold the LF
 // only when the remaining length has exactly one extra byte. This preserves
 // a legitimate first sample of 10 after a single CR delimiter.
 if(magic=="P5"&&separator=='\r'&&in.peek()=='\n') {
  auto start=in.tellg();in.seekg(0,std::ios::end);auto end=in.tellg();in.seekg(start);
  auto bytes=static_cast<std::streamoff>(im.width)*im.height*(maxval>255?2:1);
  if(end-start==bytes+1)in.get();
 }
 im.pixels.resize(static_cast<std::size_t>(im.width)*im.height);
 for(auto& p:im.pixels) { int v;
  if(magic=="P2") v=integer(token(in));
  else { int hi=in.get(); if(hi<0) throw std::runtime_error("Truncated PGM: "+path); v=hi;
   if(maxval>255){int lo=in.get();if(lo<0) throw std::runtime_error("Truncated 16-bit PGM: "+path);v=hi*256+lo;} }
  if(v<0||v>maxval) throw std::runtime_error("PGM sample outside range: "+path);
  p=static_cast<float>(255.0*v/maxval);
 }
 validateImage(im);return im;
}
void savePGM(const std::string& path,const Image& image) {
 if(image.width<=0||image.height<=0||image.pixels.size()!=static_cast<std::size_t>(image.width)*image.height) throw std::runtime_error("Invalid output image");
 std::ofstream out(path,std::ios::binary);if(!out) throw std::runtime_error("Cannot write: "+path);
 out<<"P5\n"<<image.width<<' '<<image.height<<"\n255\n";
 for(float p:image.pixels){if(!std::isfinite(p)) throw std::runtime_error("Non-finite output pixel");out.put(static_cast<char>(std::lround(std::clamp(p,0.f,255.f))));}
 if(!out) throw std::runtime_error("Write failed: "+path);
}
