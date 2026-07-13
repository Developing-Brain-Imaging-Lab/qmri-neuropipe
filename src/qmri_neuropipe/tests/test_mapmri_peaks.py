import numpy as np
import pytest

from qmri_neuropipe.interfaces.dipy import _mapmri_peaks_from_fit


def test_mapmri_peaks_accepts_float32_odf_from_fit():
    pytest.importorskip("dipy")

    class Float32MapmriFit:
        def odf(self, sphere):
            odf = np.zeros((1, len(sphere.vertices)), dtype=np.float32)
            odf[0, np.argmax(sphere.vertices[:, 0])] = 1.0
            return odf

    peaks = _mapmri_peaks_from_fit(Float32MapmriFit(), n_peaks=3)

    assert peaks.shape == (1, 9)
    assert peaks.dtype == np.float32
    assert np.all(np.isfinite(peaks))
    assert np.linalg.norm(peaks[0, :3]) == pytest.approx(1.0)
