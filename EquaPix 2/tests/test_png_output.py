#!/usr/bin/env python3
import pathlib
from png_test_utils import read_png
ROOT = pathlib.Path(__file__).resolve().parents[1]
def main():
    folder = ROOT/'build/png_test'
    assert read_png(folder/'small.png') == (4,2,bytes([0,0,10,13,32,35,128,255]))
    assert read_png(folder/'tiny.png') == (1,1,bytes([75]))
    assert read_png(folder/'multiblock.png') == (256,256,bytes(range(256))*256)
    w,h,_ = read_png(ROOT/'build/test_tmp/portrait.png')
    assert (w,h) == (144,144)
    print('PASS: PNG signatures, chunk CRCs, zlib/Adler integrity, exact pixels, quantization, multiple DEFLATE blocks and PNG portrait')
if __name__ == '__main__': main()
