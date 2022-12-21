#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class BaseFrequency:
    def get_frequency(self, term: str):
        raise NotImplementedError("subclass should implement this function")
