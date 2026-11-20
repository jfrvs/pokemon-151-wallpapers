"""Reorderable sprite post-processing filters and pipeline execution."""

from typing import Any

import numpy as np
from PIL import Image, ImageFilter

PIXEL_SCALERS = frozenset({"scale2x", "hq2x", "xbr2x"})

FILTER_LABELS: dict[str, str] = {
    "scale2x": "EPX / Scale2x (45\u00b0 Bevel Scaler)",
    "hq2x": "hq2x (8-Neighbor Weighted AA)",
    "xbr2x": "xBR2x (Directional Edge Scaler)",
    "morphological": "Morphological Corner-Cutting (Mode / Median)",
    "ssaa": "Supersampling Downscale (SSAA)",
    "gaussian_blur": "Alpha-Blended Gaussian Blur",
}

RESAMPLERS: dict[str, Image.Resampling] = {
    "Bilinear": Image.Resampling.BILINEAR,
    "Bicubic": Image.Resampling.BICUBIC,
    "Lanczos": Image.Resampling.LANCZOS,
}


def default_filter_params(filter_type: str) -> dict[str, Any]:
    """Returns default parameter dictionary for a given filter type."""
    if filter_type == "scale2x":
        return {"passes": 1, "tolerance": 15}
    if filter_type == "hq2x":
        return {"passes": 1, "strength": 0.75}
    if filter_type == "xbr2x":
        return {"passes": 1, "blend": 0.65}
    if filter_type == "morphological":
        return {"mode": "ModeFilter", "kernel": 3, "upscale": 2}
    if filter_type == "ssaa":
        return {"factor": 2, "resampler": "Bilinear"}
    if filter_type == "gaussian_blur":
        return {"radius": 0.6, "strength": 0.30}
    return {}


def _yuv_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes perceptual weighted YUV distance between RGB arrays."""
    diff = a - b
    dr, dg, db = diff[..., 0], diff[..., 1], diff[..., 2]
    dy = np.abs(0.299 * dr + 0.587 * dg + 0.114 * db)
    du = np.abs(-0.147 * dr - 0.289 * dg + 0.436 * db)
    dv = np.abs(0.615 * dr - 0.515 * dg - 0.100 * db)
    return 48.0 * dy + 7.0 * du + 6.0 * dv


def _similar(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    """Checks whether two float32 RGB arrays are within L1 tolerance."""
    return np.sum(np.abs(a - b), axis=-1) <= tol


def _scale2x_pass(arr: np.ndarray, tol: float) -> np.ndarray:
    """Runs a single vectorized EPX / Scale2x pass (doubles size)."""
    h, w, _ = arr.shape
    pad = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="edge")
    p = pad[1 : h + 1, 1 : w + 1]
    a = pad[0:h, 1 : w + 1]
    d = pad[2 : h + 2, 1 : w + 1]
    c = pad[1 : h + 1, 0:w]
    b = pad[1 : h + 1, 2 : w + 2]

    valid = (~_similar(c, b, tol)) & (~_similar(a, d, tol))
    m0 = (valid & _similar(c, a, tol))[..., None]
    m1 = (valid & _similar(a, b, tol))[..., None]
    m2 = (valid & _similar(d, c, tol))[..., None]
    m3 = (valid & _similar(b, d, tol))[..., None]

    out = np.empty((h * 2, w * 2, 3), dtype=np.float32)
    out[0::2, 0::2] = np.where(m0, a, p)
    out[0::2, 1::2] = np.where(m1, b, p)
    out[1::2, 0::2] = np.where(m2, c, p)
    out[1::2, 1::2] = np.where(m3, d, p)
    return out


def apply_scale2x(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    """Applies 1 or 2 passes of EPX / Scale2x to smooth 45-degree edges."""
    passes = int(params.get("passes", 1))
    tol = float(params.get("tolerance", 15))
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    for _ in range(passes):
        arr = _scale2x_pass(arr, tol)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _hq2x_pass(arr: np.ndarray, strength: float) -> np.ndarray:
    """Runs a vectorized 8-neighbor weighted hq2x-style magnification pass."""
    h, w, _ = arr.shape
    pad = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="edge")
    c = pad[1 : h + 1, 1 : w + 1]
    n = pad[0:h, 1 : w + 1]
    s = pad[2 : h + 2, 1 : w + 1]
    w_px = pad[1 : h + 1, 0:w]
    e_px = pad[1 : h + 1, 2 : w + 2]

    thresh = 1800.0
    same_ns = _yuv_dist(n, s) <= thresh
    same_we = _yuv_dist(w_px, e_px) <= thresh
    diag_ok = (~same_ns) & (~same_we)

    def _corner(v_adj: np.ndarray, h_adj: np.ndarray) -> np.ndarray:
        match = (diag_ok & (_yuv_dist(v_adj, h_adj) <= thresh))[..., None]
        avg = 0.5 * (v_adj + h_adj)
        blended = (1.0 - strength) * c + strength * avg
        return np.where(match, blended, c)

    out = np.empty((h * 2, w * 2, 3), dtype=np.float32)
    out[0::2, 0::2] = _corner(n, w_px)
    out[0::2, 1::2] = _corner(n, e_px)
    out[1::2, 0::2] = _corner(s, w_px)
    out[1::2, 1::2] = _corner(s, e_px)
    return out


def apply_hq2x(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    """Applies 1 or 2 passes of 8-neighbor weighted hq2x smoothing."""
    passes = int(params.get("passes", 1))
    strength = float(params.get("strength", 0.75))
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    for _ in range(passes):
        arr = _hq2x_pass(arr, strength)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _xbr_br_corner(pad: np.ndarray, blend: float) -> np.ndarray:
    """Computes bottom-right subpixel via 5x5 directional xBR edge rules."""
    h = pad.shape[0] - 4
    w = pad.shape[1] - 4

    def at(dy: int, dx: int) -> np.ndarray:
        return pad[2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w]

    e = at(0, 0)
    b = at(-1, 0)
    c = at(-1, 1)
    d = at(0, -1)
    f = at(0, 1)
    f4 = at(0, 2)
    g = at(1, -1)
    h_px = at(1, 0)
    i_px = at(1, 1)
    i4 = at(1, 2)
    h5 = at(2, 0)
    i5 = at(2, 1)

    e_w = (
        _yuv_dist(e, c)
        + _yuv_dist(e, g)
        + _yuv_dist(i_px, h5)
        + _yuv_dist(i_px, f4)
        + 4.0 * _yuv_dist(h_px, f)
    )
    i_w = (
        _yuv_dist(h_px, d)
        + _yuv_dist(h_px, i5)
        + _yuv_dist(f, i4)
        + _yuv_dist(f, b)
        + 4.0 * _yuv_dist(e, i_px)
    )

    d_ef = _yuv_dist(e, f)
    d_eh = _yuv_dist(e, h_px)
    is_edge = (e_w < i_w) & ~((d_ef <= 1200.0) & (d_eh <= 1200.0))

    px = np.where((d_ef <= d_eh)[..., None], f, h_px)
    blended = e * (1.0 - blend) + px * blend
    return np.where(is_edge[..., None], blended, e)


def _xbr2x_pass(arr: np.ndarray, blend: float) -> np.ndarray:
    """Runs a single vectorized xBR2x pass across all 4 rotated corners."""
    h, w, _ = arr.shape
    pad = np.pad(arr, ((2, 2), (2, 2), (0, 0)), mode="edge")

    e3 = _xbr_br_corner(pad, blend)
    e2 = np.rot90(_xbr_br_corner(np.rot90(pad, 1), blend), 3)
    e0 = np.rot90(_xbr_br_corner(np.rot90(pad, 2), blend), 2)
    e1 = np.rot90(_xbr_br_corner(np.rot90(pad, 3), blend), 1)

    out = np.empty((h * 2, w * 2, 3), dtype=np.float32)
    out[0::2, 0::2] = e0
    out[0::2, 1::2] = e1
    out[1::2, 0::2] = e2
    out[1::2, 1::2] = e3
    return out


def apply_xbr2x(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    """Applies 1 or 2 passes of directional xBR2x edge smoothing."""
    passes = int(params.get("passes", 1))
    blend = float(params.get("blend", 0.65))
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    for _ in range(passes):
        arr = _xbr2x_pass(arr, blend)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_morphological(
    img: Image.Image,
    params: dict[str, Any],
) -> Image.Image:
    """Rounds pixel corners using upscaled ModeFilter or MedianFilter."""
    mode = str(params.get("mode", "ModeFilter"))
    kernel = int(params.get("kernel", 3))
    upscale = int(params.get("upscale", 2))

    work = img.convert("RGB")
    if upscale > 1:
        up_size = (work.width * upscale, work.height * upscale)
        work = work.resize(up_size, Image.Resampling.NEAREST)

    if mode == "MedianFilter":
        return work.filter(ImageFilter.MedianFilter(size=kernel))
    return work.filter(ImageFilter.ModeFilter(size=kernel))


def apply_ssaa(
    img: Image.Image,
    params: dict[str, Any],
    target_w: int,
) -> Image.Image:
    """Applies supersampling downscale using the selected resampling kernel."""
    factor = int(params.get("factor", 2))
    resamp_name = str(params.get("resampler", "Bilinear"))
    resampler = RESAMPLERS.get(resamp_name, Image.Resampling.BILINEAR)

    work = img.convert("RGB")
    out_w = max(work.width, target_w)
    hi_w = out_w * factor
    work = work.resize((hi_w, hi_w), Image.Resampling.NEAREST)
    return work.resize((out_w, out_w), resampler)


def apply_gaussian_blur(
    img: Image.Image,
    params: dict[str, Any],
) -> Image.Image:
    """Applies an alpha-blended Gaussian blur to soften sharp edges."""
    radius = float(params.get("radius", 0.6))
    strength = float(params.get("strength", 0.30))
    if radius <= 0.0 or strength <= 0.0:
        return img
    rgb = img.convert("RGB")
    blurred = rgb.filter(ImageFilter.GaussianBlur(radius=radius))
    alpha = max(0.0, min(1.0, strength))
    return Image.blend(rgb, blurred, alpha=alpha)


def run_filter_pipeline(
    base_sprite: Image.Image,
    target_px: int,
    bg_rgb: tuple[int, int, int],
    pipeline: list[dict[str, Any]],
) -> Image.Image:
    """Runs an ordered list of filters on a padded sprite and fits to size."""
    active = [step for step in pipeline if step.get("enabled", True)]
    if not active:
        return base_sprite.resize(
            (target_px, target_px),
            Image.Resampling.NEAREST,
        )

    pad = 2
    base_w = base_sprite.width
    padded_base = base_w + 2 * pad
    work = Image.new("RGB", (padded_base, padded_base), bg_rgb)
    work.paste(base_sprite.convert("RGB"), (pad, pad))

    padded_target = int(round(target_px * (padded_base / base_w)))

    for idx, step in enumerate(active):
        f_type = str(step["type"])
        params: dict[str, Any] = step.get("params", {})
        upcoming_scalers = any(
            str(s["type"]) in PIXEL_SCALERS for s in active[idx + 1 :]
        )

        if f_type == "scale2x":
            work = apply_scale2x(work, params)
        elif f_type == "hq2x":
            work = apply_hq2x(work, params)
        elif f_type == "xbr2x":
            work = apply_xbr2x(work, params)
        elif f_type == "morphological":
            work = apply_morphological(work, params)
        elif f_type == "ssaa":
            dst_w = work.width if upcoming_scalers else padded_target
            work = apply_ssaa(work, params, target_w=dst_w)
        elif f_type == "gaussian_blur":
            if not upcoming_scalers and work.width < padded_target:
                resamp = (
                    Image.Resampling.NEAREST
                    if work.width == padded_base
                    else Image.Resampling.LANCZOS
                )
                work = work.resize((padded_target, padded_target), resamp)
            work = apply_gaussian_blur(work, params)

    crop_pad = int(round(work.width * (pad / padded_base)))
    cropped = work.crop(
        (crop_pad, crop_pad, work.width - crop_pad, work.height - crop_pad)
    )

    if cropped.width != target_px:
        resamp = (
            Image.Resampling.LANCZOS
            if cropped.width > target_px
            else Image.Resampling.BICUBIC
        )
        cropped = cropped.resize((target_px, target_px), resamp)

    return cropped
