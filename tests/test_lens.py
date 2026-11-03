"""The lens from what the phone says about its camera (dfpipeline/retarget/lens.py)."""

import math

import numpy as np
import pytest

from dfpipeline.retarget import lens

# A typical main camera: 4000x3000 sensor, 6.4 x 4.8 mm, 5.2 mm lens.
MAIN = {
    "focal_lengths_mm": [5.2],
    "sensor_size_mm": [6.4, 4.8],
    "pixel_array": [4000, 3000],
    "active_array": [0, 0, 4000, 3000],
}


def _camera(**over):
    camera = {"schema": 1, "lens_facing": "back", "logical": dict(MAIN), "frames": []}
    camera.update(over)
    return camera


def test_without_the_phone_the_lens_is_the_old_guess():
    got = lens.resolve(None, 1280, 720)
    assert got.source == "guess"
    assert got.K[0, 0] == pytest.approx(0.95 * 1280)


def test_the_focal_length_follows_from_the_lens_and_the_sensor():
    """5.2 mm over 6.4 mm of a 4000-pixel row is 3250 sensor pixels; the
    16:9 video keeps the sensor's whole width, scaled to 1280: 1040."""
    got = lens.resolve(_camera(), 1280, 720)
    assert got.source == "phone"
    assert got.K[0, 0] == pytest.approx(3250 * 1280 / 4000)
    assert got.K[1, 1] == pytest.approx(3250 * 1280 / 4000)   # square pixels stay square
    assert (got.K[0, 2], got.K[1, 2]) == (640, 360)
    assert got.hfov_deg == pytest.approx(math.degrees(2 * math.atan(640 / 1040)), abs=0.01)


def test_a_portrait_video_is_the_sensor_turned():
    got = lens.resolve(_camera(), 720, 1280)
    assert got.K[0, 0] == pytest.approx(1040)
    assert (got.K[0, 2], got.K[1, 2]) == (360, 640)


def test_zoom_is_a_narrower_crop_of_the_sensor():
    zoomed = lens.resolve(_camera(frames=[{"crop": [1000, 750, 3000, 2250]}] * 3), 1280, 720)
    assert zoomed.K[0, 0] == pytest.approx(2 * 1040)
    by_ratio = lens.resolve(_camera(frames=[{"zoom_ratio": 2.0}]), 1280, 720)
    assert by_ratio.K[0, 0] == pytest.approx(2 * 1040)


def test_the_lens_in_use_is_the_physical_camera_the_frames_name():
    """A phone's 0.5x is a separate lens behind one logical camera."""
    wide = dict(MAIN, focal_lengths_mm=[2.2])
    camera = _camera(physical={"2": wide}, frames=[{"physical_id": "2", "focal_mm": 2.2}] * 4)
    assert lens.resolve(camera, 1280, 720).K[0, 0] == pytest.approx(2.2 * 4000 / 6.4 * 1280 / 4000)


def test_a_physical_lens_is_read_whole_not_through_the_logical_crop():
    """The logical camera's crop and zoom are in its own pixels; the 0.5x
    lens behind it, at the widest zoom the app records at, is used whole."""
    wide = dict(MAIN, focal_lengths_mm=[2.2], pixel_array=[3000, 2250], active_array=[0, 0, 3000, 2250],
                sensor_size_mm=[4.8, 3.6])
    frames = [{"physical_id": "2", "focal_mm": 2.2, "crop": [0, 0, 4000, 3000], "zoom_ratio": 0.6}] * 3
    got = lens.resolve(_camera(physical={"2": wide}, frames=frames), 1280, 720)
    assert got.K[0, 0] == pytest.approx(2.2 * 3000 / 4.8 * 1280 / 3000)


def test_a_factory_calibration_wins_and_zeros_do_not_count():
    calibrated = dict(MAIN, intrinsics=[3300.0, 3300.0, 2000.0, 1500.0, 0.0])
    got = lens.resolve(_camera(logical=calibrated), 1280, 720)
    assert got.source == "phone-calibration"
    assert got.K[0, 0] == pytest.approx(3300 * 1280 / 4000)
    zeros = dict(MAIN, intrinsics=[0, 0, 0, 0, 0])
    assert lens.resolve(_camera(logical=zeros), 1280, 720).source == "phone"


@pytest.mark.parametrize("broken", [
    {"logical": {}},                                                    # nothing about the sensor
    {"logical": dict(MAIN, sensor_size_mm=[64.0, 48.0])},               # a 7° lens: not a phone's
    {"logical": dict(MAIN, active_array=[0, 0, 0, 0])},
    {"logical": dict(MAIN, focal_lengths_mm=["x"])},
])
def test_data_that_makes_no_sense_falls_back_to_the_guess(broken):
    assert lens.resolve(_camera(**broken), 1280, 720).source == "guess"


def test_stabilisation_is_noted():
    assert lens.resolve(_camera(frames=[{"stabilization": 1}]), 1280, 720).stabilized


def test_a_field_of_view_makes_a_lens():
    got = lens.from_hfov(60.0, 1280, 720, "depth-model")
    assert got.hfov_deg == pytest.approx(60.0)
    assert lens.from_hfov(5.0, 1280, 720, "depth-model") is None


# ------------------------------------------------------------------- gravity

def _gravity(x, y, z, rotation=0):
    return _camera(display_rotation=rotation, start_ns=0,
                   gravity=[{"t_ns": i * 10**8, "x": x, "y": y, "z": z} for i in range(5)])


def test_a_phone_upright_in_portrait_sees_up_at_the_top_of_the_picture():
    """Held upright, the sensor reads +9.8 along the device's y (up the
    screen); the picture's y runs down, so up is -y."""
    np.testing.assert_allclose(lens.up_in_image(_gravity(0, 9.8, 0)), [0, -1, 0], atol=1e-9)


def test_a_phone_looking_down_at_a_table_sees_up_behind_it():
    """Screen up, camera down: up is out of the screen, against the camera's view."""
    np.testing.assert_allclose(lens.up_in_image(_gravity(0, 0, 9.8)), [0, 0, -1], atol=1e-9)


def test_in_landscape_the_axes_turn_with_the_display():
    # ROTATION_90: the display's x is the device's y (Android's remap table).
    np.testing.assert_allclose(lens.up_in_image(_gravity(-9.8, 0, 0, rotation=1)), [0, -1, 0], atol=1e-9)
    np.testing.assert_allclose(lens.up_in_image(_gravity(9.8, 0, 0, rotation=3)), [0, -1, 0], atol=1e-9)


def test_no_gravity_or_the_front_camera_says_nothing():
    assert lens.up_in_image(None) is None
    assert lens.up_in_image(_camera()) is None
    assert lens.up_in_image(dict(_gravity(0, 9.8, 0), lens_facing="front")) is None
    assert lens.up_in_image(_gravity(0, 0.5, 0)) is None          # not gravity
