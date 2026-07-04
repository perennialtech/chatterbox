class ChatterboxVCError(RuntimeError):
    pass


class VoiceConditioningError(ChatterboxVCError):
    pass


class BackendUnavailableError(ChatterboxVCError):
    pass
