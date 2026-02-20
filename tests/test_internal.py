# test_internal.py
import numpy as np

def test_compute_de_matches_gaussian_formula():
    import main as m
    rng = np.random.RandomState(0)
    x = rng.randn(4000) * 3.0
    de = m.compute_DE(x)
    expected = 0.5 * np.log(2 * np.pi * np.e * (3.0 ** 2))
    assert np.isfinite(de)
    assert np.isclose(de, expected, rtol=0.15)

def test_compute_de_per_band_runs():
    import main as m
    x = np.random.randn(1000)  # 10 s at 100 Hz
    bands = m.compute_DE_per_band(x, 100, m.compute_DE)
    assert len(bands) == 4

def test_extract_file_id_variants():
    import main as m
    assert m.extract_file_id("DE_P01.mat") == "P01"
    assert m.extract_file_id("abc_xyz_123.mat") == "123"
    assert isinstance(m.extract_file_id(""), str)
