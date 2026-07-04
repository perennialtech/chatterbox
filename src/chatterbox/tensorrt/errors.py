class TensorRTError(RuntimeError):
    pass


class TensorRTBuildError(TensorRTError):
    pass


class TensorRTRuntimeError(TensorRTError):
    pass


class TensorRTShapeError(TensorRTRuntimeError):
    pass
