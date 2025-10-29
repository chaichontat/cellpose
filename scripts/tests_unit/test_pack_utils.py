import os
import sys
import types
import unittest
import importlib.util
import numpy as np


def _load_pack_utils():
    # Load the helper module by file path to avoid importing the top-level
    # cellpose package (which requires torch and other deps).
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, os.pardir))
    mod_path = os.path.join(root, "cellpose", "contrib", "pack_utils.py")
    spec = importlib.util.spec_from_file_location("pack_utils", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot locate module at {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pack_utils = _load_pack_utils()
compute_stripe_layout = pack_utils.compute_stripe_layout
pack_planes_to_stripes = pack_utils.pack_planes_to_stripes
unpack_stripes_to_planes = pack_utils.unpack_stripes_to_planes


class TestPackUtils(unittest.TestCase):
    def test_compute_layout_three_stripes(self):
        Ly = 68
        layout = compute_stripe_layout(Ly, bsize=256, pack_k=3, guard=16)
        self.assertIsNotNone(layout)
        self.assertEqual(layout.K, 3)
        self.assertEqual(layout.guard, 16)
        self.assertEqual(layout.starts, (0, 84, 168))
        self.assertEqual(layout.stripe_height, 68)
        self.assertEqual(layout.bsize, 256)

    def test_pack_unpack_identity(self):
        Lz, Ly, Lx, C = 7, 68, 113, 3
        x = np.random.RandomState(0).randn(Lz, Ly, Lx, C).astype(np.float32)
        layout = compute_stripe_layout(Ly, bsize=256, pack_k=3, guard=16)
        self.assertIsNotNone(layout)
        packed, mapping = pack_planes_to_stripes(x, layout)
        # Create injective head
        y_packed = np.concatenate([packed[..., :1], 2 * packed[..., 1:2], 3 * packed[..., 2:3]], axis=-1)
        y = unpack_stripes_to_planes(y_packed, mapping, Lz=Lz, Ly=Ly)
        x0 = np.concatenate([x[..., :1], 2 * x[..., 1:2], 3 * x[..., 2:3]], axis=-1)
        np.testing.assert_allclose(y, x0, rtol=0, atol=0)

    def test_pack_unpack_remainder_two(self):
        Lz, Ly, Lx, C = 5, 68, 77, 1
        x = np.random.RandomState(1).randn(Lz, Ly, Lx, C).astype(np.float32)
        layout = compute_stripe_layout(Ly, bsize=256, pack_k=3, guard=16)
        self.assertIsNotNone(layout)
        self.assertEqual(layout.K, 3)
        packed, mapping = pack_planes_to_stripes(x, layout)
        y_packed = np.tile(packed, (1, 1, 1, 3))
        y = unpack_stripes_to_planes(y_packed, mapping, Lz=Lz, Ly=Ly)
        expected = np.tile(x, (1, 1, 1, 3))
        np.testing.assert_allclose(y, expected, rtol=0, atol=0)

    def test_pack_layout_insufficient_height(self):
        Ly = 200
        layout = compute_stripe_layout(Ly, bsize=256, pack_k=3, guard=60)
        self.assertIsNone(layout)

    def test_pack_unpack_exact_multiple(self):
        Lz, Ly, Lx, C = 6, 68, 41, 2
        x = np.random.RandomState(2).randn(Lz, Ly, Lx, C).astype(np.float32)
        layout = compute_stripe_layout(Ly, bsize=256, pack_k=3, guard=16)
        self.assertIsNotNone(layout)
        self.assertEqual(layout.K, 3)
        packed, mapping = pack_planes_to_stripes(x, layout)
        y_packed = packed[..., :1].repeat(3, axis=-1)
        y = unpack_stripes_to_planes(y_packed, mapping, Lz=Lz, Ly=Ly)
        expected = x[..., :1].repeat(3, axis=-1)
        np.testing.assert_allclose(y, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
