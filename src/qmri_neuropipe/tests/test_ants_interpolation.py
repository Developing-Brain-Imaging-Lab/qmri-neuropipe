from qmri_neuropipe.interfaces.ants import _normalize_interpolator


def test_generic_interpolation_aliases_are_valid_ants_names():
    assert _normalize_interpolator("sinc") == "lanczosWindowedSinc"
    assert _normalize_interpolator("cubic") == "bSpline"
    assert _normalize_interpolator("nearest") == "nearestNeighbor"
    assert _normalize_interpolator("trilinear") == "linear"


def test_canonical_ants_interpolation_name_is_preserved():
    assert _normalize_interpolator("welchWindowedSinc") == "welchWindowedSinc"
