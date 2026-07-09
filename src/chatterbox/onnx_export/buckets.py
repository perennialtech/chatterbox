from __future__ import annotations

from ..models.s3gen.pipeline import _TOKEN_LENGTH_BUCKETS

TOKEN_TO_MU_TOKEN_BUCKETS = tuple(int(x) for x in _TOKEN_LENGTH_BUCKETS)

FLOW_MEL_BUCKETS = tuple(
    sorted({63, 64, 65, *(int(x) * 2 for x in TOKEN_TO_MU_TOKEN_BUCKETS)})
)

VOCODER_MEL_BUCKETS = tuple(
    sorted({1, 64, *(int(x) * 2 for x in TOKEN_TO_MU_TOKEN_BUCKETS)})
)
