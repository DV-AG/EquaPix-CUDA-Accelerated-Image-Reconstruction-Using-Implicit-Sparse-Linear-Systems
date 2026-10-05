#!/usr/bin/env python3
"""Real CUDA smoke tests. Requires a compiled equapix and an NVIDIA GPU."""
import csv
import pathlib
import subprocess
import tempfile
from png_test_utils import read_png
ROOT = pathlib.Path(__file__).resolve().parents[1]
def pixels(path):
    return read_png(path)[2]
def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp); dataset = tmp/'input'
        subprocess.run(['python3', str(ROOT/'scripts/generate_dataset.py'), '--output', str(dataset), '--count', '3', '--width', '24', '--height', '20'], check=True)
        cases = [('random', '0', '1'), ('random', '100', '1001'), ('random', '30', '500'), ('block', '20', '501')]
        for index, (mode, percent, iterations) in enumerate(cases):
            out = tmp/f'run{index}'
            subprocess.run([str(ROOT/'equapix'), '--input', str(dataset), '--output', str(out), '--mask', mode, '--missing-percent', percent, '--block-size', '9', '--iterations', iterations, '--cpu-benchmark-count', '3'], check=True)
            with (out/'metrics.csv').open() as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 3
            assert not list(out.rglob('*.pgm')), 'result images must all be PNG'
            for image in out.rglob('*.png'):
                read_png(image)
            for row in rows:
                assert row['validation'] == 'PASS'
                assert float(row['cpu_gpu_max_abs']) <= .002
                assert int(row['iterations']) == int(iterations)
                name = pathlib.Path(row['image_name']).stem
                original = pixels(out/'examples'/f'{name}_original.png')
                result = pixels(out/'examples'/f'{name}_reconstructed.png')
                for y in range(20):
                    for x in range(24):
                        if x in (0,23) or y in (0,19):
                            assert result[y*24+x] == original[y*24+x]
                if mode == 'random' and percent == '0':
                    assert result == original
                    assert float(row['rmse']) == 0
    print('PASS: real CUDA/CPU comparisons, odd/even iteration counts, 0/100% masks, block mask, fixed borders')
if __name__ == '__main__': main()
