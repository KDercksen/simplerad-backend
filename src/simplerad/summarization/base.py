#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class BaseSummarizer:
    def summarize(self, text: str):
        raise NotImplementedError("subclass should implement this function")
