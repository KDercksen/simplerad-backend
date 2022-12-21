#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..utils import LazyValueDict, preprocess
from .embedding import TransformerFrequency

frequencizers = LazyValueDict(
    {
        "transformer": TransformerFrequency,
    }
)


def get_frequencies(text: str):
    preprocessed = preprocess(text)
    estimated_freq, sample_size = frequencizers.get_model().get_frequency(
        preprocessed["text"]
    )
    # TODO: properly implement this new version
    return {
        "global_frequency": estimated_freq,
        "global_certainty": sample_size,
        "local_frequency": estimated_freq,
        "local_certainty": sample_size,
    }


__all__ = [frequencizers, get_frequencies]
