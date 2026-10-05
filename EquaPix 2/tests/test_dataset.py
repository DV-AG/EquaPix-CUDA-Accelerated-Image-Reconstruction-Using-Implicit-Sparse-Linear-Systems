import pathlib
import subprocess
import tempfile
import unittest
import hashlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/generate_dataset.py'
class DatasetTests(unittest.TestCase):
    def test_generator_exists(self):
        self.assertTrue(SCRIPT.is_file(), 'dataset generator is required')
    @unittest.skipUnless(SCRIPT.is_file(), 'implementation missing')
    def test_deterministic_varied_p5_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = pathlib.Path(tmp)/'a', pathlib.Path(tmp)/'b'
            for dst in (a, b):
                subprocess.run(['python3', str(SCRIPT), '--output', str(dst), '--count', '12', '--width', '16', '--height', '12', '--seed', '77'], check=True, capture_output=True)
            first = sorted(a.glob('*.pgm')); second = sorted(b.glob('*.pgm'))
            self.assertEqual(len(first), 12)
            hashes = set()
            for p, q in zip(first, second):
                data = p.read_bytes(); self.assertEqual(data, q.read_bytes())
                magic, dims, maxval, pixels = data.split(b'\n', 3)
                self.assertEqual((magic, dims, maxval), (b'P5', b'16 12', b'255'))
                self.assertEqual(len(pixels), 192)
                hashes.add(hashlib.sha256(pixels).digest())
            self.assertGreaterEqual(len(hashes), 9)
            bad = subprocess.run(['python3', str(SCRIPT), '--count', '0'], capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
if __name__ == '__main__': unittest.main()
