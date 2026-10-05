// Small native PNG encoder for EquaPix's 8-bit grayscale result images.
// PNG uses filter 0 and a zlib stream of stored DEFLATE blocks. This trades
// compression for zero external dependencies; no Python/conversion is used.
#include "image_io.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
namespace {
using Bytes=std::vector<unsigned char>;
void bigEndian(Bytes& out,std::uint32_t value){for(int shift=24;shift>=0;shift-=8)out.push_back(static_cast<unsigned char>(value>>shift));}
std::uint32_t crc32(const Bytes& data){
 static const auto table=[] {std::array<std::uint32_t,256> t{};
  for(std::uint32_t i=0;i<256;++i){auto c=i;for(int k=0;k<8;++k)c=(c>>1)^((c&1)?0xedb88320u:0);t[i]=c;}return t;}();
 std::uint32_t crc=0xffffffffu;for(auto b:data)crc=table[(crc^b)&255]^(crc>>8);return crc^0xffffffffu;
}
void chunk(std::ofstream& out,const char* type,const Bytes& data){
 if(data.size()>0x7fffffffu)throw std::runtime_error("PNG chunk too large");
 Bytes header;bigEndian(header,static_cast<std::uint32_t>(data.size()));
 out.write(reinterpret_cast<const char*>(header.data()),4);
 Bytes checksum_data(type,type+4);checksum_data.insert(checksum_data.end(),data.begin(),data.end());
 out.write(reinterpret_cast<const char*>(checksum_data.data()),static_cast<std::streamsize>(checksum_data.size()));
 Bytes checksum;bigEndian(checksum,crc32(checksum_data));out.write(reinterpret_cast<const char*>(checksum.data()),4);
}
Bytes storedZlib(const Bytes& raster){
 Bytes out{0x78,0x01}; // DEFLATE, 32 KiB window, fastest/no compression.
 for(std::size_t offset=0;offset<raster.size();){
  auto n=static_cast<std::uint16_t>(std::min<std::size_t>(65535,raster.size()-offset));
  bool last=offset+n==raster.size();out.push_back(last?1:0); // BFINAL and stored BTYPE; byte aligned.
  out.push_back(static_cast<unsigned char>(n));out.push_back(static_cast<unsigned char>(n>>8));
  auto inverse=static_cast<std::uint16_t>(~n);out.push_back(static_cast<unsigned char>(inverse));out.push_back(static_cast<unsigned char>(inverse>>8));
  out.insert(out.end(),raster.begin()+offset,raster.begin()+offset+n);offset+=n;
 }
 std::uint32_t a=1,b=0;for(auto value:raster){a=(a+value)%65521;b=(b+a)%65521;}
 bigEndian(out,(b<<16)|a);return out;
}
}
std::string resultImageName(const std::string& input_name,const std::string& kind){
 return std::filesystem::path(input_name).stem().string()+"_"+kind+".png";
}
void savePNG(const std::string& path,const Image& image){
 if(image.width<=0||image.height<=0||static_cast<long long>(image.width)*image.height>std::numeric_limits<int>::max()||image.pixels.size()!=static_cast<std::size_t>(image.width)*image.height)
  throw std::runtime_error("Invalid PNG image dimensions or pixel count");
 auto row_size=static_cast<std::size_t>(image.width)+1;
 if(row_size*image.height>0x7f000000u)throw std::runtime_error("PNG image too large for this lightweight encoder");
 Bytes raster;raster.reserve(row_size*image.height);
 for(int y=0;y<image.height;++y){raster.push_back(0); // PNG filter type 0: unfiltered grayscale scanline.
  for(int x=0;x<image.width;++x){float value=image.pixels[static_cast<std::size_t>(y)*image.width+x];
   if(!std::isfinite(value))throw std::runtime_error("Non-finite PNG pixel");
   raster.push_back(static_cast<unsigned char>(std::lround(std::clamp(value,0.f,255.f))));
  }
 }
 Bytes header;bigEndian(header,static_cast<std::uint32_t>(image.width));bigEndian(header,static_cast<std::uint32_t>(image.height));
 header.insert(header.end(),{8,0,0,0,0}); // 8 bits, grayscale, standard compression/filter, no interlace.
 auto compressed=storedZlib(raster);
 std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("Cannot write PNG: "+path);
 const unsigned char signature[]{137,80,78,71,13,10,26,10};out.write(reinterpret_cast<const char*>(signature),8);
 chunk(out,"IHDR",header);chunk(out,"IDAT",compressed);chunk(out,"IEND",{});
 out.flush();if(!out)throw std::runtime_error("PNG write failed: "+path);
}
