#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class BaseSearcher:
    def search(self, text: str):
        raise NotImplementedError("subclass should implement this function")


class BaseTwoStageSearcher(BaseSearcher):
    def rank(self, query_terms, indexes):
        raise NotImplementedError("subclass should implement this function")
