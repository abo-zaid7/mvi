
import os
from dataclasses import dataclass, field, fields, replace
from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import frangi, threshold_otsu, apply_hysteresis_threshold
from skimage.morphology import (remove_small_objects, remove_small_holes, disk,
                                skeletonize)
from skimage.measure import regionprops

cv2.setNumThreads(1)

@dataclass
class PipelineParams:

    size: int = 1024

    scale_reference: int = 1024
    use_scaled_sigmas: bool = True

    use_clahe: bool = True
    use_shade_correction: bool = True
    use_bcosfire: bool = True
    use_frangi: bool = True
    use_fov_mask: bool = True
    use_od_mask: bool = True

    od_mode: str = "statistics"
    use_nlm: bool = False
    use_shape_filter: bool = False

    od_detect_mode: str = "brightest"

    od_suppress_gain: float = 0.0

    clahe_clip: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)

    shade_median_frac: float = 0.06

    nlm_h: float = 0.6
    nlm_patch: int = 5
    nlm_distance: int = 6

    bcosfire_sigma: float = 4.8
    bcosfire_rho: Tuple[float, ...] = (0, 6, 12, 18, 24)
    bcosfire_sigma0: float = 0.4
    bcosfire_alpha: float = 0.1
    bcosfire_rotations: int = 12
    bcosfire_thin_sigma: float = 2.8
    bcosfire_thin_rho: Tuple[float, ...] = (0, 4, 8, 12)
    bcosfire_thin_weight: float = 0.0

    use_bcosfire_bank: bool = False
    bcosfire_bank_sigmas: Tuple[float, ...] = (2.2, 4.8, 7.5)
    bcosfire_bank_weights: Tuple[float, ...] = ()
    bcosfire_bank_combine: str = "max"

    frangi_sigmas: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
    frangi_beta: float = 0.5
    frangi_gamma: float = 0.02

    fuse_weight_bcosfire: float = 0.8

    fov_threshold: float = 0.06
    fov_erode_frac: float = 0.012
    od_radius_frac: float = 0.055
    od_smooth_frac: float = 0.04

    thresh_mode: str = "percentile"
    thresh_high: float = 97.0
    thresh_low: float = 86.0

    adapt_floor_pct: float = 70.0
    adapt_shrink: float = 0.6
    adapt_prior_high: float = 97.0
    adapt_low_offset: float = 11.0
    adapt_min_pct: float = 92.0
    adapt_max_pct: float = 99.3

    adapt_shift: float = 0.0

    baseline_frangi_sigma: float = 4.0
    baseline_frangi_thresh: float = 92.0
    baseline_bcosfire_sigma: float = 4.8
    baseline_bcosfire_thresh: float = 88.0

    shape_max_width: float = 14.0
    shape_max_solidity: float = 0.62
    shape_min_elong: float = 1.6
    shape_min_skel_ratio: float = 0.055
    shape_exempt_area: int = 60
    shape_keep_main_tree: bool = True

    min_object_px: int = 150
    close_radius: int = 1
    fill_hole_px: int = 12

DEFAULT_PARAMS = PipelineParams()

OPERATING_POINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "operating_point.json")

def load_operating_point(path=None, **overrides):

    import json
    path = path or OPERATING_POINT_PATH
    with open(path) as f:
        d = json.load(f)
    d.pop("_provenance", None)
    valid = {f.name for f in fields(PipelineParams)}
    kw = {k: (tuple(v) if isinstance(v, list) else v)
          for k, v in d.items() if k in valid}
    kw.update(overrides)
    return PipelineParams(**kw)

def scale_factor(params):
    if not params.use_scaled_sigmas:
        return 1.0
    return float(params.size) / float(params.scale_reference)

def _len(value, params):
    return value * scale_factor(params)

def _area(value, params):
    return value * scale_factor(params) ** 2

def green_channel(rgb):

    return rgb[..., 1].astype(np.float32)

def invert(image):

    return 1.0 - image

def apply_clahe(image, params=DEFAULT_PARAMS):
    u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=params.clahe_clip,
                            tileGridSize=params.clahe_grid)
    return clahe.apply(u8).astype(np.float32) / 255.0

def shade_correct(image, params=DEFAULT_PARAMS):

    k = int(params.shade_median_frac * image.shape[1]) | 1
    k = max(3, min(k, 255))
    u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    background = cv2.medianBlur(u8, k).astype(np.float32) / 255.0
    corrected = image - background
    return corrected

def denoise_nlm(image, params=DEFAULT_PARAMS):

    from skimage.restoration import denoise_nl_means, estimate_sigma
    sigma = float(np.mean(estimate_sigma(image)))
    return denoise_nl_means(image, h=params.nlm_h * sigma, fast_mode=True,
                            patch_size=params.nlm_patch,
                            patch_distance=params.nlm_distance)

def preprocess(rgb, params=DEFAULT_PARAMS):

    img = invert(green_channel(rgb))
    if params.use_clahe:
        img = apply_clahe(img, params)
    if params.use_nlm:
        img = denoise_nlm(img, params)
    if params.use_shade_correction:
        img = shade_correct(img, params)
    return img.astype(np.float32)

def fov_mask(rgb, params=DEFAULT_PARAMS):

    bright = rgb.max(axis=2)
    m = bright > params.fov_threshold
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    r = max(1, int(params.fov_erode_frac * rgb.shape[1]))
    m = cv2.erode(m.astype(np.uint8), disk(r).astype(np.uint8)).astype(bool)
    return m

def optic_disc_mask(rgb, fov, params=DEFAULT_PARAMS):

    sm = int(params.od_smooth_frac * rgb.shape[1]) | 1
    lum = (rgb[..., 0] + rgb[..., 1]) / 2.0
    lum = np.where(fov, lum, 0.0)
    blurred = cv2.GaussianBlur(lum, (sm, sm), 0)
    blurred = np.where(fov, blurred, -1.0)

    if params.od_detect_mode == "robust":
        cy, cx = _od_centre_robust(rgb, fov, blurred, params)
    else:
        cy, cx = np.unravel_index(int(np.argmax(blurred)), blurred.shape)

    r = int(params.od_radius_frac * rgb.shape[1])
    yy, xx = np.ogrid[:rgb.shape[0], :rgb.shape[1]]
    od = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    return od, (cy, cx, r)

def _od_centre_robust(rgb, fov, blurred, params):

    g = 1.0 - rgb[..., 1].astype(np.float32)
    ridge = _dog(g, _len(6.0, params))
    k = int(params.od_radius_frac * rgb.shape[1] * 2) | 1
    density = cv2.GaussianBlur(ridge, (k, k), 0)
    density = np.where(fov, density, 0.0)

    def nz(x):
        x = np.where(fov, x, x[fov].min() if fov.any() else 0.0)
        lo, hi = np.percentile(x[fov], 1), np.percentile(x[fov], 99.9)
        return np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1)

    score = np.where(fov, nz(blurred) + nz(density), -1.0)
    return np.unravel_index(int(np.argmax(score)), score.shape)

def _dog(image, sigma, ratio=2.0):

    a = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma * 0.5)
    b = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma * 0.5 * ratio)
    return np.maximum(a - b, 0.0)

def _bcosfire_single(image, sigma, rho_list, params):

    dog_maps = {}
    for rho in rho_list:
        d = _dog(image, sigma)
        sig_blur = _len(params.bcosfire_sigma0, params) + params.bcosfire_alpha * rho
        dog_maps[rho] = cv2.GaussianBlur(d, (0, 0), sigmaX=max(sig_blur, 0.3))

    rho_max = max(rho_list) if max(rho_list) > 0 else 1.0
    weights = {rho: float(np.exp(-(rho ** 2) / (2 * (rho_max ** 2 + 1e-8))))
               for rho in rho_list}

    h, w = image.shape
    best = np.zeros_like(image)

    for k in range(params.bcosfire_rotations):
        psi = np.pi * k / params.bcosfire_rotations
        log_acc = np.zeros_like(image)
        wsum = 0.0
        for rho in rho_list:
            phis = [psi] if rho == 0 else [psi, psi + np.pi]
            for phi in phis:
                dx = -rho * np.cos(phi)
                dy = -rho * np.sin(phi)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(dog_maps[rho], M, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0)
                log_acc += weights[rho] * np.log(shifted + 1e-6)
                wsum += weights[rho]
        resp = np.exp(log_acc / max(wsum, 1e-8))
        best = np.maximum(best, resp)

    best = best - best.min()
    return best.astype(np.float32)

def bcosfire_bank(image, params=DEFAULT_PARAMS):

    base_rho = np.array(params.bcosfire_rho, dtype=float)
    maps = []
    for sigma in params.bcosfire_bank_sigmas:
        rho = tuple(base_rho * (sigma / params.bcosfire_sigma))
        r = _bcosfire_single(image, _len(sigma, params),
                             tuple(_len(x, params) for x in rho), params)
        maps.append(_norm(r))
    if not maps:
        return np.zeros_like(image)
    if params.bcosfire_bank_combine == "max":
        return np.maximum.reduce(maps).astype(np.float32)
    w = np.array(params.bcosfire_bank_weights or [1.0] * len(maps), dtype=np.float32)
    w = w[:len(maps)] / w[:len(maps)].sum()
    return sum(wi * m for wi, m in zip(w, maps)).astype(np.float32)

def bcosfire(image, params=DEFAULT_PARAMS):

    if params.use_bcosfire_bank:
        return bcosfire_bank(image, params)
    thick = _bcosfire_single(image, _len(params.bcosfire_sigma, params),
                             tuple(_len(x, params) for x in params.bcosfire_rho),
                             params)
    if params.bcosfire_thin_weight <= 0:
        return thick
    thin = _bcosfire_single(image, _len(params.bcosfire_thin_sigma, params),
                            tuple(_len(x, params) for x in params.bcosfire_thin_rho),
                            params)
    w = params.bcosfire_thin_weight
    return ((1 - w) * _norm(thick) + w * _norm(thin)).astype(np.float32)

def frangi_multiscale(image, params=DEFAULT_PARAMS):

    r = frangi(image, sigmas=[_len(x, params) for x in params.frangi_sigmas],
               black_ridges=False,
               beta=params.frangi_beta, gamma=params.frangi_gamma)
    return np.nan_to_num(r).astype(np.float32)

def _norm(x, mask=None):

    v = x[mask] if mask is not None and mask.any() else x
    hi = np.percentile(v, 99.5)
    if hi <= 0:
        hi = float(v.max()) or 1.0
    return np.clip(x / hi, 0, 1).astype(np.float32)

def enhance(image, valid=None, params=DEFAULT_PARAMS):

    maps, weights = [], []
    if params.use_bcosfire:
        maps.append(_norm(bcosfire(image, params), valid))
        weights.append(params.fuse_weight_bcosfire)
    if params.use_frangi:
        maps.append(_norm(frangi_multiscale(image, params), valid))
        weights.append(1.0 - params.fuse_weight_bcosfire)
    if not maps:
        return _norm(image, valid)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()
    out = sum(w * m for w, m in zip(weights, maps))
    return _norm(out, valid)

def adaptive_percentiles(vesselness, valid, params=DEFAULT_PARAMS):

    v = vesselness[valid] if valid is not None and valid.any() else vesselness.ravel()
    v = v[np.isfinite(v)]
    prior = float(params.adapt_prior_high)
    if v.size < 1000:
        return prior, prior - params.adapt_low_offset

    if v.size > 200_000:
        v = v[:: max(1, v.size // 200_000)]
    vs = np.sort(v)

    floor = float(np.percentile(vs, params.adapt_floor_pct))
    tail = vs[vs > floor]
    p_otsu = prior
    if tail.size >= 256 and float(tail[-1] - tail[0]) > 1e-6:
        try:
            t = float(threshold_otsu(tail))
            p_otsu = 100.0 * np.searchsorted(vs, t) / vs.size
        except Exception:
            p_otsu = prior

    k = float(np.clip(params.adapt_shrink, 0.0, 1.0))
    p_high = float(np.clip(k * p_otsu + (1.0 - k) * prior + params.adapt_shift,
                           params.adapt_min_pct, params.adapt_max_pct))
    p_low = max(1.0, p_high - float(params.adapt_low_offset))
    return p_high, p_low

def resolve_thresholds(vesselness, valid, params=DEFAULT_PARAMS):

    if params.thresh_mode == "percentile":
        v = vesselness[valid] if valid is not None and valid.any() else vesselness
        hi = float(np.percentile(v, params.thresh_high))
        lo = float(np.percentile(v, params.thresh_low))
        return hi, lo
    if params.thresh_mode == "adaptive":
        v = vesselness[valid] if valid is not None and valid.any() else vesselness
        p_hi, p_lo = adaptive_percentiles(vesselness, valid, params)
        return float(np.percentile(v, p_hi)), float(np.percentile(v, p_lo))
    return float(params.thresh_high), float(params.thresh_low)

def hysteresis(vesselness, params=DEFAULT_PARAMS, valid=None):

    hi, lo = resolve_thresholds(vesselness, valid, params)
    if hi <= lo:
        hi = lo + 1e-6
    return apply_hysteresis_threshold(vesselness, lo, hi)

def shape_filter(mask, params=DEFAULT_PARAMS, return_rejected=False):

    m = mask.astype(bool)
    if not params.use_shape_filter or not m.any():
        return (m, np.zeros_like(m)) if return_rejected else m

    struct8 = np.ones((3, 3), dtype=int)
    lab, n = ndi.label(m, structure=struct8)
    if n == 0:
        return (m, np.zeros_like(m)) if return_rejected else m

    idx = np.arange(1, n + 1)
    area = np.asarray(ndi.sum(m, lab, idx), dtype=np.float64)
    skel = skeletonize(m)
    slen = np.maximum(np.asarray(ndi.sum(skel, lab, idx), dtype=np.float64), 1.0)

    width = area / slen
    skel_ratio = slen / np.maximum(area, 1.0)

    solidity = np.ones(n, dtype=np.float64)
    elong = np.full(n, 99.0, dtype=np.float64)
    for r in regionprops(lab):
        i = r.label - 1
        solidity[i] = r.solidity
        minor = max(r.minor_axis_length, 1e-6)
        elong[i] = r.major_axis_length / minor

    exempt = area < _area(params.shape_exempt_area, params)
    too_wide = width > _len(params.shape_max_width, params)
    too_thin_skel = skel_ratio < params.shape_min_skel_ratio / max(scale_factor(params), 1e-8)
    blobby = (solidity > params.shape_max_solidity) & (elong < params.shape_min_elong)

    reject = (too_wide | too_thin_skel | blobby) & ~exempt
    if params.shape_keep_main_tree:
        reject[int(np.argmax(area))] = False

    keep_labels = np.zeros(n + 1, dtype=bool)
    keep_labels[1:] = ~reject
    kept = keep_labels[lab]
    if return_rejected:
        return kept, m & ~kept
    return kept

def cleanup(mask, params=DEFAULT_PARAMS):

    m = mask.astype(bool)
    r = max(0, int(round(_len(params.close_radius, params))))
    if r > 0:
        se = disk(r).astype(np.uint8)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, se).astype(bool)
    if params.fill_hole_px > 0:
        m = remove_small_holes(m, area_threshold=int(_area(params.fill_hole_px, params)))
    if params.min_object_px > 0:
        m = remove_small_objects(m, min_size=max(2, int(_area(params.min_object_px, params))))
    return m

def compute_context(rgb, params=DEFAULT_PARAMS):

    pre = preprocess(rgb, params)

    fov = fov_mask(rgb, params) if params.use_fov_mask else np.ones(rgb.shape[:2], bool)

    if params.use_od_mask:
        od, od_geom = optic_disc_mask(rgb, fov, params)
    else:
        od, od_geom = np.zeros(rgb.shape[:2], bool), None

    if params.use_od_mask and params.od_mode in ("delete", "statistics", "suppress"):
        stat = fov & ~od
    else:
        stat = fov
    valid = fov & ~od if (params.use_od_mask and params.od_mode == "delete") else fov

    ves = enhance(pre, stat, params)
    if params.use_od_mask and params.od_mode == "suppress" and params.od_suppress_gain > 0:

        g = float(np.clip(params.od_suppress_gain, 0.0, 1.0))
        ves = np.where(od, ves * (1.0 - g), ves)
    ves = np.where(valid, ves, 0.0).astype(np.float32)

    return {"pre": pre, "fov": fov, "od": od, "od_geom": od_geom,
            "stat": stat, "valid": valid, "vesselness": ves}

def segment_from_context(ctx, params=DEFAULT_PARAMS):

    m = hysteresis(ctx["vesselness"], params, ctx.get("stat", ctx["valid"]))
    m = m & ctx["valid"]
    m = shape_filter(m, params)
    return cleanup(m, params)

def segment(rgb, params=DEFAULT_PARAMS):

    return segment_from_context(compute_context(rgb, params), params)

def baseline_otsu(rgb, params=DEFAULT_PARAMS, ctx=None):

    pre = ctx["pre"] if ctx is not None else preprocess(rgb, params)
    fov = ctx["fov"] if ctx is not None else fov_mask(rgb, params)
    vals = pre[fov]
    t = threshold_otsu(vals)
    m = (pre > t) & fov
    return cleanup(m, params)

def baseline_frangi_response(rgb, params=DEFAULT_PARAMS, ctx=None):

    pre = ctx["pre"] if ctx is not None else preprocess(rgb, params)
    r = frangi(pre, sigmas=[_len(params.baseline_frangi_sigma, params)],
               black_ridges=False,
               beta=params.frangi_beta, gamma=params.frangi_gamma)
    return _norm(np.nan_to_num(r))

def baseline_bcosfire_response(rgb, params=DEFAULT_PARAMS, ctx=None):

    pre = ctx["pre"] if ctx is not None else preprocess(rgb, params)
    fov = ctx["fov"] if ctx is not None else fov_mask(rgb, params)
    bp = replace(params, bcosfire_sigma=params.baseline_bcosfire_sigma,
                 use_bcosfire_bank=False, bcosfire_thin_weight=0.0)
    return _norm(bcosfire(pre, bp), fov), fov

def baseline_single_scale_frangi(rgb, params=DEFAULT_PARAMS, ctx=None, response=None):

    r = response if response is not None else baseline_frangi_response(rgb, params, ctx)
    m = r > np.percentile(r, params.baseline_frangi_thresh)
    return cleanup(m, params)

def baseline_bcosfire(rgb, params=DEFAULT_PARAMS, ctx=None, response=None):

    if response is not None:
        r, fov = response
    else:
        r, fov = baseline_bcosfire_response(rgb, params, ctx)
    m = (r > np.percentile(r[fov], params.baseline_bcosfire_thresh)) & fov
    return cleanup(m, params)
