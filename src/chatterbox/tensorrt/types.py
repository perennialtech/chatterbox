from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeRange:
    min: tuple[int, ...]
    opt: tuple[int, ...]
    max: tuple[int, ...]

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "min": list(self.min),
            "opt": list(self.opt),
            "max": list(self.max),
        }
