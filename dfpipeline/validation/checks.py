"""Tier 1: deterministic checks. No model, no tokens, no cost.

These exist to throw out the clips that are obviously unusable — a black
screen, a frozen frame, a two-second misfire — before we spend anything asking
a model about them. Each check is a small pure function over `ClipFacts`, so
adding a new rule is one function plus one registry entry.
"""

from __future__ import annotations

from collections.abc import Callable

from PIL import Image, ImageChops, ImageStat

from dfpipeline.validation.schemas import ClipFacts, Finding, Thresholds

Check = Callable[[ClipFacts, Thresholds], Finding | None]

REGISTRY: list[Check] = []


def check(fn: Check) -> Check:
    REGISTRY.append(fn)
    return fn


@check
def undecodable(facts: ClipFacts, _: Thresholds) -> Finding | None:
    if not facts.decodable or facts.frame_count == 0:
        return Finding("undecodable", "This clip could not be played back — the file looks corrupt or empty.")
    return None


@check
def too_short(facts: ClipFacts, t: Thresholds) -> Finding | None:
    if facts.duration is not None and facts.duration < t.min_duration:
        return Finding(
            "too_short",
            f"The clip is only {facts.duration:.1f}s long; a demonstration needs at least {t.min_duration:.0f}s.",
        )
    return None


@check
def too_long(facts: ClipFacts, t: Thresholds) -> Finding | None:
    if facts.duration is not None and facts.duration > t.max_duration:
        return Finding("too_long", f"The clip runs {facts.duration:.0f}s, longer than the {t.max_duration:.0f}s limit.")
    return None


@check
def low_resolution(facts: ClipFacts, t: Thresholds) -> Finding | None:
    if facts.width is not None and facts.width < t.min_width:
        return Finding("low_resolution", f"The video is only {facts.width}px wide — too low to be usable.")
    return None


@check
def black_screen(facts: ClipFacts, t: Thresholds) -> Finding | None:
    if facts.mean_luma is not None and facts.mean_luma < t.black_luma:
        return Finding("black_screen", "The picture is black — the camera was covered or did not start.")
    return None


@check
def no_motion(facts: ClipFacts, t: Thresholds) -> Finding | None:
    if facts.motion is not None and facts.motion < t.min_motion:
        return Finding("no_motion", "Nothing moves in this clip — no demonstration was performed.")
    return None


def run_checks(facts: ClipFacts, thresholds: Thresholds) -> list[Finding]:
    """All failing rules, so the operator sees every problem at once."""
    return [f for rule in REGISTRY if (f := rule(facts, thresholds)) is not None]


def measure(frames: list[tuple[float, Image.Image]]) -> tuple[float | None, float | None]:
    """Mean brightness and frame-to-frame motion from the sampled frames.

    Pillow only — bringing in numpy/OpenCV for two scalars would not earn its
    place in the API image.
    """
    if not frames:
        return None, None
    greys = [img.convert("L") for _, img in frames]
    luma = sum(ImageStat.Stat(g).mean[0] for g in greys) / len(greys)
    if len(greys) < 2:
        return luma, None
    diffs = [ImageStat.Stat(ImageChops.difference(a, b)).mean[0] for a, b in zip(greys, greys[1:])]
    return luma, sum(diffs) / len(diffs)
