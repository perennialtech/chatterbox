class OnnxExportError(RuntimeError):
    pass


class OnnxValidationError(OnnxExportError):
    pass


class OnnxRuntimeError(RuntimeError):
    pass
