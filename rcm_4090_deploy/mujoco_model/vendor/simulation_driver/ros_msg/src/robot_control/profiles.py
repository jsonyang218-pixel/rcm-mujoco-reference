"""Explicit needle variants used by the shared monitor shell."""

from dataclasses import dataclass

from .kinematics import (
    DEFAULT_TIP_OFFSET_MM,
    LEGACY_NEEDLE_OFFSET_SIGN,
    NEW_NEEDLE_OFFSET_SIGN,
    NEW_NEEDLE_SOLID_LENGTH_MM,
)


@dataclass(frozen=True)
class NeedleProfile:
    """Geometry that differs between the two physical needle variants."""

    name: str
    offset_sign: float
    tip_offset_mm: float = DEFAULT_TIP_OFFSET_MM
    solid_length_mm: float = 0.0
    # Presentation metadata; target planning is shared by both variants.
    show_junction_trajectory: bool = True

    @property
    def is_replacement(self):
        return self.offset_sign < 0.0


LEGACY_NEEDLE_PROFILE = NeedleProfile(
    name="legacy",
    offset_sign=LEGACY_NEEDLE_OFFSET_SIGN,
    show_junction_trajectory=True,
)

NEW_NEEDLE_PROFILE = NeedleProfile(
    name="new",
    offset_sign=NEW_NEEDLE_OFFSET_SIGN,
    solid_length_mm=NEW_NEEDLE_SOLID_LENGTH_MM,
    show_junction_trajectory=False,
)
