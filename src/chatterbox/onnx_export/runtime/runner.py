from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OnnxGraphRunner:
    name: str
    session: object
    input_names: list[str]
    output_names: list[str]

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        feed = {}
        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"{self.name}: missing input {name}")
            feed[name] = np.ascontiguousarray(inputs[name])
        outputs = self.session.run(self.output_names, feed)
        return {name: value for name, value in zip(self.output_names, outputs)}
