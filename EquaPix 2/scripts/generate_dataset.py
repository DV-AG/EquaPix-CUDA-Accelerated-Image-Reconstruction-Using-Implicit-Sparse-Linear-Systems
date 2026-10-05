#!/usr/bin/env python3
"""Deterministic varied P5 grayscale dataset; Python standard library only."""
import argparse
import math
import pathlib
import random
import json
PATTERNS = ('gradient', 'circles', 'rectangles', 'checkerboard', 'sine', 'diagonal', 'combined', 'smooth', 'texture')
def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return n

def generate_pixels(index, width, height, seed):
    rng = random.Random(seed + index * 1000003)
    phase = rng.uniform(0, 2 * math.pi)
    frequency = rng.uniform(2, 6)
    kind = index % len(PATTERNS)
    pixels = bytearray(width * height)
    for y in range(height):
        v = y / max(height - 1, 1)
        for x in range(width):
            u = x / max(width - 1, 1)
            r = math.hypot(u - .5, v - .5)
            if kind == 0:
                value = 20 + 210 * (.65*u + .35*v)
            elif kind == 1:
                value = 220 if r < .22 else (120 if r < .38 else 25)
            elif kind == 2:
                value = 180 if .15 < u < .75 and .2 < v < .65 else 30
                if .5 < u < .9 and .5 < v < .85:
                    value = 245
            elif kind == 3:
                value = 220 if (int(u*10) + int(v*10)) % 2 else 35
            elif kind == 4:
                value = 128 + 100*math.sin(frequency*math.pi*u + phase)*math.cos(2*math.pi*v)
            elif kind == 5:
                value = 200 if int((u+v)*9 + phase) % 2 else 50
            elif kind == 6:
                value = 30 + 90*u + 60*v
                if r < .25:
                    value += 65
                if .1 < u < .35 and .15 < v < .8:
                    value += 40
            elif kind == 7:
                value = 40 + 185*math.exp(-12*((u-.45)**2+(v-.55)**2)) + 15*math.sin(2*math.pi*u + phase)
            else:
                value = 70 + 90*u + 35*math.sin(12*math.pi*v + phase) + rng.uniform(-30, 30)
            pixels[y*width+x] = max(0, min(255, round(value)))
    return pixels

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=positive, default=200)
    parser.add_argument('--width', type=positive, default=256)
    parser.add_argument('--height', type=positive, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('data/generated'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        name = f'image_{i:04d}_{PATTERNS[i % len(PATTERNS)]}.pgm'
        pixels = generate_pixels(i, args.width, args.height, args.seed)
        (args.output / name).write_bytes(f'P5\n{args.width} {args.height}\n255\n'.encode('ascii') + pixels)
    (args.output / 'manifest.json').write_text(json.dumps({'count': args.count, 'width': args.width, 'height': args.height, 'seed': args.seed, 'patterns': PATTERNS}, indent=2) + '\n')
    print(f'Generated {args.count} P5 images ({args.width}x{args.height}), seed={args.seed}, output={args.output}')
if __name__ == '__main__':
    main()
