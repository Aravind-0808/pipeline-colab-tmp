"""The camera's lens, from what the phone says about itself — else a guess.

Every distance the pipeline measures runs through the focal length. The hand
is placed by matching its known size to its size in pixels, so its depth is
`f * size / pixels`: a focal length 20% off puts the hand 20% too far or too
near, and every robot position after it inherits that. Until now the focal
length was a constant guess, `0.95 * max(width, height)`, for every phone.

The Android app now sends, with each clip, what the camera reports of itself
(`camera` in the track.hand spec; CONTRACT.md): the lens's focal length in mm,
the sensor's physical size and pixel grid, the part of the sensor the video
was read from (zoom and crop), and, on the few phones that have it, a factory
calibration. From those the focal length in the video's own pixels follows.
They are kept raw on the phone and turned into numbers here, so a mistake is
fixed with a pipeline deploy, not an app release.

Also from the phone: which way is up — the gravity sensor, turned into the
picture's axes — for the table plane (scene.py).

No phone data, or data that makes no sense (a field of view no phone camera
has), and the old guess is used; the bundle records which (`lens_source`).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from statistics import median

import numpy as np

log = logging.getLogger("hand2robot.lens")

#: The guess used when the phone said nothing: fx = fy = GUESS * max(w, h).
GUESS = 0.95

# Horizontal fields of view a phone's video can plausibly have, in degrees:
# from a 2x telephoto to the widest ultra-wide. Outside it the data is wrong.
MIN_HFOV_DEG, MAX_HFOV_DEG = 25.0, 130.0


@dataclass(frozen=True)
class Lens:
    K: np.ndarray            # (3, 3), the video's upright pixels
    source: str              # "phone-calibration" | "phone" | "depth-model" | "guess"
    stabilized: bool = False  # video stabilisation was on: it crops a little more, unknown how much

    @property
    def hfov_deg(self) -> float:
        width = 2.0 * float(self.K[0, 2])
        return math.degrees(2.0 * math.atan(width / (2.0 * float(self.K[0, 0]))))


def from_hfov(hfov_deg: float, width: int, height: int, source: str) -> Lens | None:
    """A lens of this horizontal field of view (square pixels, centred)."""
    if not MIN_HFOV_DEG <= hfov_deg <= MAX_HFOV_DEG:
        return None
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return Lens(_k(f, f, width / 2.0, height / 2.0), source)


def guess(width: int, height: int) -> Lens:
    f = GUESS * max(width, height)
    return Lens(_k(f, f, width / 2.0, height / 2.0), "guess")


def _k(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def resolve(camera: dict | None, width: int, height: int) -> Lens:
    """The lens for a video of `width` x `height` (as decoded: upright)."""
    if camera:
        try:
            lens = from_phone(camera, width, height)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            log.warning("the phone's camera data could not be read (%s); guessing the lens", exc)
            lens = None
        if lens is not None:
            return lens
    return guess(width, height)


def _num(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _sensor(camera: dict) -> tuple[dict, bool]:
    """The characteristics of the camera the video was read from, and whether
    it is a physical lens behind a logical camera: the one most frames name (a
    phone's 0.5x is often a separate lens behind one logical camera), else the
    logical camera itself."""
    frames = camera.get("frames") or []
    ids = [str(f["physical_id"]) for f in frames if f.get("physical_id") not in (None, "")]
    physical = camera.get("physical") or {}
    if ids:
        most = max(set(ids), key=ids.count)
        if most in physical:
            return physical[most], True
    return camera.get("logical") or {}, False


def from_phone(camera: dict, width: int, height: int) -> Lens | None:
    """The focal length in the video's pixels, from the phone's own figures.

    The chain: the lens's focal length in mm over the sensor's width in mm
    gives it in sensor pixels; the video is the crop region (zoom) of the
    sensor, cut to the video's aspect ratio and scaled to its size; and a
    portrait video is the landscape sensor turned a quarter.
    """
    sensor, behind_logical = _sensor(camera)
    frames = camera.get("frames") or []
    pixel_array = sensor.get("pixel_array") or []
    active = sensor.get("active_array") or []
    if len(pixel_array) != 2 or len(active) != 4:
        return None
    active_w, active_h = active[2] - active[0], active[3] - active[1]
    if active_w <= 0 or active_h <= 0:
        return None

    calibrated = _calibration(sensor)
    if calibrated is not None:
        fx_s, fy_s = calibrated
        source = "phone-calibration"
    else:
        focal = median(v for f in frames if (v := _num(f.get("focal_mm"))) and v > 0) \
            if any(_num(f.get("focal_mm")) for f in frames) else None
        if focal is None:
            focal = next((v for v in map(_num, sensor.get("focal_lengths_mm") or []) if v and v > 0), None)
        size = sensor.get("sensor_size_mm") or []
        if focal is None or len(size) != 2 or not (_num(size[0]) and _num(size[1])):
            return None
        # The physical size covers the whole pixel array (Camera2's
        # SENSOR_INFO_PHYSICAL_SIZE), so the pixel pitch is size / array.
        fx_s = focal * pixel_array[0] / float(size[0])
        fy_s = focal * pixel_array[1] / float(size[1])
        source = "phone"

    # What part of the sensor the video was read from (active-array pixels).
    # A physical lens behind a logical camera: the crop region and zoom are in
    # the logical camera's terms, not this lens's. The app records at the
    # widest zoom (CameraOverlay), which on such a phone is that whole lens.
    crop_w, crop_h = (active_w, active_h) if behind_logical else _crop(frames, active_w, active_h)

    # The video in the sensor's own orientation (sensors are landscape).
    landscape = width >= height
    out_w, out_h = (width, height) if landscape else (height, width)
    # The crop is cut to the video's aspect ratio about its centre, then scaled:
    # whichever side is the tighter fit sets the scale.
    scale = max(out_w / crop_w, out_h / crop_h)
    fx, fy = fx_s * scale, fy_s * scale
    if not landscape:
        fx, fy = fy, fx

    lens = Lens(_k(fx, fy, width / 2.0, height / 2.0), source,
                stabilized=any(bool(f.get("stabilization")) for f in frames))
    if not MIN_HFOV_DEG <= lens.hfov_deg <= MAX_HFOV_DEG:
        log.warning("the phone's camera data gives a %.0f° field of view; guessing the lens instead",
                    lens.hfov_deg)
        return None
    return lens


def _calibration(sensor: dict) -> tuple[float, float] | None:
    """Camera2's LENS_INTRINSIC_CALIBRATION [fx, fy, cx, cy, s], in the active
    array's pixels — on the few phones that report it, and only when it is not
    all zeros (some phones report zeros: a Huawei P30 Pro, for one)."""
    values = [_num(v) for v in (sensor.get("intrinsics") or [])]
    if len(values) < 2 or values[0] is None or values[1] is None:
        return None
    if values[0] <= 1.0 or values[1] <= 1.0:
        return None
    return values[0], values[1]


def _crop(frames: list[dict], active_w: float, active_h: float) -> tuple[float, float]:
    """The crop region's size, the median over the clip; a zoom ratio alone
    means a centred crop of 1/zoom; neither, the whole active array."""
    crops = [f["crop"] for f in frames if isinstance(f.get("crop"), list) and len(f["crop"]) == 4]
    if crops:
        w = median(c[2] - c[0] for c in crops)
        h = median(c[3] - c[1] for c in crops)
        if w > 0 and h > 0:
            return float(w), float(h)
    zooms = [z for f in frames if (z := _num(f.get("zoom_ratio"))) and z > 0]
    zoom = median(zooms) if zooms else 1.0
    # Zoom below 1 (0.5x) is a wider physical lens, already chosen in _sensor;
    # on that lens the crop is the whole array.
    zoom = max(zoom, 1.0)
    return active_w / zoom, active_h / zoom


# ------------------------------------------------------------------- gravity

# Android's display rotation (Surface.ROTATION_*), in quarter turns, and the
# device's axes (x right, y up, z out of the screen) seen on that display.
_DISPLAY = {
    0: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
    1: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float),
    2: np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float),
    3: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float),
}


def up_in_image(camera: dict | None, seconds: float = 1.0) -> np.ndarray | None:
    """Which way is up, as a unit vector in the picture's camera axes
    (x right, y down, z into the scene), or None.

    From the phone's gravity sensor over the first `seconds` of the take. At
    rest it reads +9.8 along the axis pointing away from the ground, so its
    direction is up. The device's axes are turned into the display's by the
    rotation the video was recorded at, then into the back camera's: the
    picture's y runs down the screen, and the camera looks out of the back,
    away from the screen. A front camera is not used for this (it records the
    operator, not the table).
    """
    if not camera or (camera.get("lens_facing") or "back") != "back":
        return None
    samples = camera.get("gravity") or []
    start = _num(camera.get("start_ns"))
    picked = []
    for s in samples:
        t = _num(s.get("t_ns"))
        if start is not None and t is not None and t - start > seconds * 1e9:
            continue
        v = [_num(s.get(k)) for k in ("x", "y", "z")]
        if None not in v:
            picked.append(v)
    if not picked:
        return None
    device_up = np.mean(np.asarray(picked, dtype=float), axis=0)
    norm = float(np.linalg.norm(device_up))
    if norm < 5.0:          # not gravity (a phone in free fall, or a bad sensor)
        return None
    rotation = int(camera.get("display_rotation") or 0) % 4
    display = _DISPLAY[rotation] @ (device_up / norm)
    up = np.array([display[0], -display[1], -display[2]])
    return up / np.linalg.norm(up)
