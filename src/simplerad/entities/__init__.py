#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from omegaconf import DictConfig

from ..utils import LazyValueDict, preprocess
from .fuzzy import SimstringPredictor
from .neural import FlairPredictor

entity_taggers = LazyValueDict(
    {
        "flair": FlairPredictor,
        "simstring": SimstringPredictor,
    }
)


def get_entities(text: str):
    preprocessed = preprocess(text)
    preprocessed["spans"] = entity_taggers.get_model().predict(preprocessed)

    return preprocessed


__all__ = [entity_taggers, get_entities]
