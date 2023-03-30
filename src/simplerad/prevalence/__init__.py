#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..utils import LazyValueDict, preprocess
from .sklearn_models import SKLearnPrevalence

prevalencers = LazyValueDict(
    {
        "global_sklearn": SKLearnPrevalence,
    }
)


def get_prevalences(text: str):
    preprocessed = preprocess(text)
    global_prevalence, global_certainty = prevalencers.get_model().get_prevalence(
        preprocessed["text"]
    )
    # TODO: properly implement this new version
    return {
        "global_prevalence": global_prevalence,
        "global_certainty": global_certainty,
    }


__all__ = [prevalencers, get_prevalences]
