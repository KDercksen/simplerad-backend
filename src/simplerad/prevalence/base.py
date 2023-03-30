#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class BasePrevalence:
    def get_prevalence(self, term: str):
        raise NotImplementedError("subclass should implement this function")
