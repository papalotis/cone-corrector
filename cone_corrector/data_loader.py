"""Generate data for the cone corrector model."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict

from cone_corrector.common_types import FloatArrayNx2, FloatArrayNx3, FloatVector, FloatVector2, IntVector


class Layout(BaseModel):
    """The layout of cones."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    x: FloatVector
    y: FloatVector
    color: IntVector

    start_position: FloatVector2
    start_orientation: float

    timing_line_position: FloatVector2
    timing_line_orientation: float
    timing_line_width: float

    @classmethod
    def from_layout_json_file(cls, path: str | Path) -> Self:
        """Load a layout from a JSON file."""
        data = json.loads(Path(path).read_text())

        return cls(**data)

    @cached_property
    def cone_positions(self) -> FloatArrayNx2:
        """Get the positions of the cones."""
        return np.column_stack((self.x, self.y))

    @cached_property
    def cones_with_color(self) -> FloatArrayNx3:
        """Get the positions of the cones with color."""
        return np.column_stack((self.x, self.y, self.color))


if __name__ == "__main__":
    path = Path(__file__).parent.parent / "layout-merchant" / "layouts" / "fsg19.json"

    layout = Layout.from_layout_json_file(path)
