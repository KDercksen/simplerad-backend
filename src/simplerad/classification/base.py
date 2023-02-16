#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List


class BaseTextClassifier:
    def predict(self, text):
        raise NotImplementedError("subclass needs to implement this function")


class BaseSentenceClassifier:
    def predict(self, text):
        raise NotImplementedError("subclass needs to implement this function")
