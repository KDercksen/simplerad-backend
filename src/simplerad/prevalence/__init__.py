#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..utils import LazyValueDict, preprocess
from .sklearn_models import SKLearnPrevalence
from .transformer_models import GlobalLocalAdapterPrevalence

prevalencers = LazyValueDict(
    {
        "global_sklearn": SKLearnPrevalence,
        "global_local_adapter": GlobalLocalAdapterPrevalence,
    }
)


def get_global_prevalence(text: str):
    preprocessed = preprocess(text)
    prevalence, certainty = prevalencers.get_model().get_global_prevalence(
        preprocessed["text"]
    )

    return {
        "prevalence": prevalence,
        "certainty": certainty,
    }


def get_local_prevalence(text: str, context: str):
    preprocessed = preprocess(text)
    preprocessed_context = preprocess(context)
    prevalence, certainty = prevalencers.get_model().get_local_prevalence(
        preprocessed["text"], preprocessed_context["text"]
    )

    return {
        "prevalence": prevalence,
        "certainty": certainty,
    }


__all__ = [prevalencers, get_global_prevalence, get_local_prevalence]
