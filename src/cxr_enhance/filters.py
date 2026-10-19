"""Auditable grayscale enhancement with explicit historical behavior."""

from pathlib import Path

import cv2
import numpy as np

METHODS = ("identity", "legacy_unsharp", "corrected_unsharp", "gaussian_hef", "ideal_hef")


def read_image(path: str | Path) -> np.ndarray:
    """Read grayscale PNG pixels without changing integer depth or intensity.

    Official Shenzhen images can store the same grayscale plane three times.
    Collapse exactly identical color channels without a luminance conversion;
    reject genuine color and alpha-channel images rather than guessing intent.
    """
    source = Path(path)
    try:
        encoded = np.fromfile(source, dtype=np.uint8)
        if encoded[:8].tobytes() != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Cannot decode image: expected PNG content in {source}")
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except (OSError, cv2.error) as error:
        raise ValueError(f"Cannot decode image: {source}") from error
    if image is None:
        raise ValueError(f"Cannot decode image: {source}")
    if image.ndim == 3 and image.shape[2] == 3:
        if not (np.array_equal(image[:, :, 0], image[:, :, 1])
                and np.array_equal(image[:, :, 0], image[:, :, 2])):
            raise ValueError(f"Expected grayscale pixels; color channels differ: {source}")
        image = np.ascontiguousarray(image[:, :, 0])
    if image.ndim != 2 or image.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"Expected uint8/uint16 grayscale image: {source}")
    if min(image.shape) < 2:
        raise ValueError(f"Image dimensions are too small: {source}")
    return image


def normalize_image(image: np.ndarray, reject_constant: bool = True) -> np.ndarray:
    """Normalize each image independently; no dataset statistics are fitted."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or min(values.shape, default=0) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite two-dimensional image of at least 2 x 2 pixels")
    lower, upper = float(values.min()), float(values.max())
    if upper <= lower:
        if reject_constant:
            raise ValueError("Constant images are invalid inputs")
        return np.zeros_like(values)
    return ((values - lower) / (upper - lower)).astype(np.float32)


def unsharp_mask(
    image: np.ndarray,
    threshold: float = 10.0,
    amount: float = 1.5,
    legacy: bool = False,
) -> np.ndarray:
    """Use normalized inputs; threshold is measured in 8-bit intensity units.

    Legacy behavior intentionally ignores the threshold, reproducing the source
    notebook's uint8 GaussianBlur/addWeighted result. Corrected behavior retains
    signed differences and suppresses residuals whose magnitude is <= threshold.
    """
    if threshold < 0 or amount < 0 or not np.isfinite([threshold, amount]).all():
        raise ValueError("Unsharp threshold and amount must be finite and nonnegative")
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or min(values.shape, default=0) < 2 or not np.isfinite(values).all():
        raise ValueError("Unsharp requires a finite two-dimensional normalized image")
    if values.min() < 0 or values.max() > 1:
        raise ValueError("Unsharp requires normalized intensity values in [0, 1]")
    if legacy:
        quantized = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(quantized, (5, 5), 1.0)
        return cv2.addWeighted(quantized, 1.0 + amount, blurred, -amount, 0).astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(values, (5, 5), 1.0)
    residual = values - blurred
    residual = np.where(np.abs(residual) > threshold / 255.0, residual, 0.0)
    return np.clip(values + amount * residual, 0, 1).astype(np.float32)


def high_frequency_emphasis(image: np.ndarray, kind: str, cutoff: float | None = None) -> np.ndarray:
    """Apply 0.5 + 1.5 * highpass, then per-image min-max normalization.

    Cutoff is in DFT bins: default (30/480)*min(height,width). Gaussian HEF
    uses the real inverse transform; ideal HEF preserves the historical magnitude
    operation. Centered fftfreq coordinates correctly handle odd image dimensions.
    """
    values = np.asarray(image, dtype=np.float32)
    if kind not in ("gaussian", "ideal"):
        raise ValueError(f"Unknown HEF kind: {kind}")
    if values.ndim != 2 or min(values.shape, default=0) < 2 or not np.isfinite(values).all():
        raise ValueError("HEF requires a finite two-dimensional image")
    cutoff = (30.0 / 480.0) * min(values.shape) if cutoff is None else float(cutoff)
    if cutoff <= 0 or not np.isfinite(cutoff):
        raise ValueError("HEF cutoff must be finite and positive")
    if float(values.max()) == float(values.min()):
        return np.zeros_like(values)
    rows, columns = values.shape
    row_bins = np.fft.fftshift(np.fft.fftfreq(rows) * rows)[:, None]
    column_bins = np.fft.fftshift(np.fft.fftfreq(columns) * columns)[None, :]
    distance_squared = row_bins**2 + column_bins**2
    if kind == "gaussian":
        highpass = 1.0 - np.exp(-distance_squared / (2.0 * cutoff**2))
    else:
        highpass = (distance_squared > cutoff**2).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(values))
    reconstructed = np.fft.ifft2(np.fft.ifftshift(spectrum * (0.5 + 1.5 * highpass)))
    enhanced = reconstructed.real if kind == "gaussian" else np.abs(reconstructed)
    return normalize_image(enhanced, reject_constant=False)


def _enhance(image: np.ndarray, method: str, legacy_geometry: bool = False) -> np.ndarray:
    if method == "identity":
        return image.astype(np.float32)
    if method in ("legacy_unsharp", "corrected_unsharp"):
        return unsharp_mask(image, legacy=method == "legacy_unsharp")
    result = high_frequency_emphasis(
        image, "gaussian" if method == "gaussian_hef" else "ideal",
        cutoff=30.0 if legacy_geometry else None,
    )
    if legacy_geometry:
        result = np.clip(result * 255, 0, 255).astype(np.uint8).astype(np.float32) / 255
    return result


def preprocess_image(
    image: np.ndarray,
    method: str,
    image_size: int = 384,
    legacy_geometry: bool = False,
) -> np.ndarray:
    """Return normalized float32 pixels; enhancement precedes zero padding.

    Modern geometry rescales the longest edge to image_size. Historical geometry
    mimics IMREAD_GRAYSCALE (including uint16 downshift), 640x480 enhancement and
    a final 224-square resize, independently of image_size.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown enhancement method: {method}")
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 8:
        raise ValueError("image_size must be an integer >= 8")
    normalized = normalize_image(image)
    if legacy_geometry:
        source = np.asarray(image)
        if source.dtype == np.uint16:
            source = (source >> 8).astype(np.uint8)
        elif source.dtype != np.uint8:
            raise ValueError("Historical geometry requires original uint8/uint16 image pixels")
        resized = cv2.resize(source, (640, 480), interpolation=cv2.INTER_LINEAR)
        enhanced = _enhance(resized.astype(np.float32) / 255.0, method, True)
        quantized = np.clip(np.rint(enhanced * 255), 0, 255).astype(np.uint8)
        return cv2.resize(quantized, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    height, width = normalized.shape
    scale = image_size / max(height, width)
    target_width = max(2, min(image_size, round(width * scale)))
    target_height = max(2, min(image_size, round(height * scale)))
    resized = cv2.resize(normalized, (target_width, target_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    resized = np.clip(resized, 0.0, 1.0)
    enhanced = _enhance(resized, method)
    padded = np.zeros((image_size, image_size), dtype=np.float32)
    top, left = (image_size - target_height) // 2, (image_size - target_width) // 2
    padded[top:top + target_height, left:left + target_width] = enhanced
    return padded
