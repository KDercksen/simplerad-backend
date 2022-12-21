#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class BasePredictor:
    def predict(self, text):
        raise NotImplementedError("subclass needs to implement this function")
