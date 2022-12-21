#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..utils import LazyValueDict, preprocess
from .abstractive import TransformerAbstractiveSummarizer

summarizers = LazyValueDict(
    {
        "transformer_abstractive": TransformerAbstractiveSummarizer,
    }
)


def get_summaries(text: str):
    preprocessed = preprocess(text)
    summary = summarizers.get_model().summarize(preprocessed["text"])
    return {"summary": summary}


__all__ = [summarizers, get_summaries]
