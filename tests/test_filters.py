from pathlib import Path

import cv2
import numpy as np
import pytest

from cxr_enhance.filters import METHODS, high_frequency_emphasis, preprocess_image, read_image, unsharp_mask


def test_read_preserves_uint16(tmp_path: Path) -> None:
    source = np.arange(120, dtype=np.uint16).reshape(10, 12) * 500
    path = tmp_path / "depth.png"
    assert cv2.imwrite(str(path), source)
    actual = read_image(path)
    assert actual.dtype == np.uint16
    np.testing.assert_array_equal(actual, source)


def test_read_rejects_color_and_corruption(tmp_path: Path) -> None:
    path = tmp_path / "color.png"
    color = np.zeros((8, 8, 3), dtype=np.uint8)
    color[2, 3, 1] = 1
    assert cv2.imwrite(str(path), color)
    with pytest.raises(ValueError, match="grayscale"):
        read_image(path)
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="decode"):
        read_image(path)
    path.write_bytes(cv2.imencode(".jpg", np.eye(8, dtype=np.uint8) * 255)[1].tobytes())
    with pytest.raises(ValueError, match="PNG"):
        read_image(path)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_read_collapses_replicated_grayscale_without_changing_pixels(tmp_path: Path, dtype) -> None:
    original = np.arange(120, dtype=dtype).reshape(10, 12)
    if dtype == np.uint16:
        original *= 500
    path = tmp_path / "replicated-grayscale.png"
    encoded = np.repeat(original[:, :, None], 3, axis=2)
    assert cv2.imwrite(str(path), encoded)
    actual = read_image(path)
    assert actual.dtype == original.dtype
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, original)
    for method in METHODS:
        np.testing.assert_array_equal(
            preprocess_image(actual, method, image_size=64),
            preprocess_image(original, method, image_size=64),
        )


def test_read_rejects_alpha_channels(tmp_path: Path) -> None:
    path = tmp_path / "alpha.png"
    assert cv2.imwrite(str(path), np.full((8, 8, 4), 255, dtype=np.uint8))
    with pytest.raises(ValueError, match="grayscale"):
        read_image(path)


def test_unsharp_threshold_and_signed_residuals() -> None:
    values = np.full((15, 15), 0.5, dtype=np.float32)
    values[7, 7] = 0.6
    values[4, 4] = 0.4
    suppressed = unsharp_mask(values, threshold=255)
    enhanced = unsharp_mask(values, threshold=0)
    np.testing.assert_array_equal(suppressed, values)
    assert enhanced[7, 7] > values[7, 7]
    assert enhanced[4, 4] < values[4, 4]
    assert not np.array_equal(unsharp_mask(values, threshold=10), suppressed)


def test_legacy_matches_historical_opencv_and_ignores_threshold() -> None:
    source = np.random.default_rng(3).integers(0, 256, (29, 31), dtype=np.uint8)
    blurred = cv2.GaussianBlur(source, (5, 5), 1.0)
    expected = cv2.addWeighted(source, 2.5, blurred, -1.5, 0).astype(np.float32) / 255
    np.testing.assert_array_equal(unsharp_mask(source / 255, threshold=0, legacy=True), expected)
    np.testing.assert_array_equal(unsharp_mask(source / 255, threshold=255, legacy=True), expected)


@pytest.mark.parametrize("method", METHODS)
def test_aspect_resize_padding_and_bounds(method: str) -> None:
    source = np.random.default_rng(4).integers(0, 65536, (23, 46), dtype=np.uint16)
    output = preprocess_image(source, method, image_size=64)
    assert output.shape == (64, 64)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert 0 <= output.min() <= output.max() <= 1
    np.testing.assert_array_equal(output[:16], 0)
    np.testing.assert_array_equal(output[48:], 0)


@pytest.mark.parametrize("kind", ["gaussian", "ideal"])
def test_hef_constant_and_odd_dimensions(kind: str) -> None:
    np.testing.assert_array_equal(high_frequency_emphasis(np.ones((13, 17)), kind), 0)
    values = np.random.default_rng(8).random((13, 17), dtype=np.float32)
    output = high_frequency_emphasis(values, kind)
    assert output.shape == values.shape
    assert output.min() == 0
    assert output.max() == 1
    with pytest.raises(ValueError, match="cutoff"):
        high_frequency_emphasis(values, kind, cutoff=0)


def test_hef_methods_are_distinct() -> None:
    values = np.random.default_rng(9).random((43, 37), dtype=np.float32)
    assert not np.allclose(high_frequency_emphasis(values, "gaussian"), high_frequency_emphasis(values, "ideal"))


@pytest.mark.parametrize("method", METHODS)
def test_resize_roundoff_stays_inside_normalized_range(method: str) -> None:
    source = np.full((9, 13), 255, dtype=np.uint8)
    source[0, 0] = 0
    output = preprocess_image(source, method, image_size=8)
    assert output.dtype == np.float32
    assert output.shape == (8, 8)
    assert np.isfinite(output).all()
    assert 0 <= output.min() <= output.max() <= 1


def test_historical_geometry_matches_original_8bit_unsharp() -> None:
    source = np.random.default_rng(7).integers(0, 65536, (30, 45), dtype=np.uint16)
    quantized = (source >> 8).astype(np.uint8)
    resized = cv2.resize(quantized, (640, 480))
    enhanced = cv2.addWeighted(resized, 2.5, cv2.GaussianBlur(resized, (5, 5), 1.0), -1.5, 0)
    expected = cv2.resize(enhanced, (224, 224)).astype(np.float32) / 255
    actual = preprocess_image(source, "legacy_unsharp", legacy_geometry=True)
    np.testing.assert_array_equal(actual, expected)


def test_preprocessing_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Constant"):
        preprocess_image(np.ones((8, 8)), "identity")
    with pytest.raises(ValueError, match="Unknown"):
        preprocess_image(np.eye(8), "unknown")
    with pytest.raises(ValueError, match="finite"):
        preprocess_image(np.full((8, 8), np.nan), "identity")
    with pytest.raises(ValueError, match="image_size"):
        preprocess_image(np.eye(8), "identity", image_size=1)


@pytest.mark.parametrize("values", [np.full((8, 8), np.nan), np.array([0, 1]), np.zeros((1, 8))])
def test_unsharp_invalid_geometry_or_values(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite"):
        unsharp_mask(values)


@pytest.mark.parametrize("values", [np.full((8, 8), -0.1), np.full((8, 8), 1.1)])
def test_unsharp_requires_normalized_input(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="normalized"):
        unsharp_mask(values)
