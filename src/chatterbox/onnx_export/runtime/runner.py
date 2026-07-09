from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OnnxGraphRunner:
    name: str
    session: object
    input_names: list[str]
    output_names: list[str]
    actual_input_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.actual_input_names and hasattr(self.session, "get_inputs"):
            self.actual_input_names = [inp.name for inp in self.session.get_inputs()]

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        required_inputs = self.actual_input_names or self.input_names
        feed = {}
        for name in required_inputs:
            if name not in inputs:
                raise KeyError(f"{self.name}: missing input {name}")
            feed[name] = np.ascontiguousarray(inputs[name])

        outputs = self.session.run(self.output_names, feed)
        return {name: value for name, value in zip(self.output_names, outputs)}
