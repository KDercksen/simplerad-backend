#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class BasePrevalence:
    def get_global_prevalence(self, term: str):
        raise NotImplementedError("subclass should implement this function")

    def get_local_prevalence(self, term: str, context: str):
        raise NotImplementedError("subclass should implement this function")
