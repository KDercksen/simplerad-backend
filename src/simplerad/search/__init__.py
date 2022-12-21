#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..utils import LazyValueDict, preprocess
from .bm25 import BM25
from .dense import FastTextFAISSJSONLFolderSearcher
from .exact import ExactJSONLFolderSearcher
from .fuzzy import SimstringJSONLFolderSearcher

searchers = LazyValueDict(
    {
        "exact": ExactJSONLFolderSearcher,
        "simstring": SimstringJSONLFolderSearcher,
        "faiss": FastTextFAISSJSONLFolderSearcher,
        "bm25": BM25,
    }
)


def get_search_results(text: str):
    preprocessed = preprocess(text)
    results = searchers.get_model().search(preprocessed["text"])
    return {"data": sorted(results, key=lambda x: x["score"], reverse=True)}


__all__ = [searchers, get_search_results]
