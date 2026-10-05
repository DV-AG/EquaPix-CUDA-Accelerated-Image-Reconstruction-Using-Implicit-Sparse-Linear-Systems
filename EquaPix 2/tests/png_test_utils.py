"""Independent stdlib decoder for the grayscale PNGs emitted by this project.
Used by tests only; execution never requires PNG conversion or a Python codec.
"""
import pathlib
import struct
import zlib

def read_png(path):
    data = pathlib.Path(path).read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n', f'Invalid PNG signature: {path}'
    offset, payload, size, ended = 8, bytearray(), None, False
    while offset < len(data):
        length = struct.unpack_from('>I', data, offset)[0]
        kind = data[offset+4:offset+8]
        body = data[offset+8:offset+8+length]
        checksum = struct.unpack_from('>I', data, offset+8+length)[0]
        assert zlib.crc32(kind+body) & 0xffffffff == checksum, f'PNG CRC mismatch: {path}'
        if kind == b'IHDR':
            width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', body)
            assert (depth, color, compression, filtering, interlace) == (8,0,0,0,0)
            size = (width, height)
        elif kind == b'IDAT':
            payload.extend(body)
        elif kind == b'IEND':
            assert length == 0
            ended = True
        offset += 12+length
    assert ended and size and offset == len(data)
    width, height = size
    raster = zlib.decompress(payload)
    assert len(raster) == (width+1)*height
    pixels = bytearray()
    for y in range(height):
        row = raster[y*(width+1):(y+1)*(width+1)]
        assert row[0] == 0
        pixels.extend(row[1:])
    return width, height, bytes(pixels)
