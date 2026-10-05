

import csv
import math
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
def check_run(path, expected, comparisons):
    with (path/'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == expected, f'{path}: expected {expected} images, got {len(rows)}'
    assert not list(path.rglob('*.pgm')), f'{path}: unexpected PGM result images'
    passed = 0
    for row in rows:
        assert math.isfinite(float(row['gpu_ms'])) and float(row['gpu_ms']) >= 0
        assert math.isfinite(float(row['rmse'])) and float(row['rmse']) >= 0
        if row['validation'] == 'PASS':
            assert float(row['cpu_gpu_max_abs']) <= .002
            passed += 1
        name = pathlib.Path(row['image_name']).stem
        for kind in ('original', 'masked', 'reconstructed'):
            image = path/'examples'/f'{name}_{kind}.png'
            assert image.is_file() and image.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
        assert (path/'matrix_portraits'/f'{name}_matrix.png').is_file()
    assert passed >= comparisons, f'{path}: insufficient CPU validations'
    log = (path/'execution_log.txt').read_text()
    assert 'CUDA device:' in log and f'Processed successfully: {expected} | failed: 0' in log
    print(f'PASS: {path.name}, {expected} images, {passed} CPU/GPU validations, complete artifacts')
def main():
    for level in (5,10,20,30):
        check_run(ROOT/'results'/f'random_{level}', 200, 10)
    check_run(ROOT/'results/block_64', 8, 8)
    print('CUDA experiment outputs verified successfully.')
if __name__ == '__main__': main()
