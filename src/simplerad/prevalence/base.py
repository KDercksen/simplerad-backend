#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class BasePrevalence:
    def get_global_prevalence(self, term: str):
        raise NotImplementedError("subclass should implement this function")

    def get_local_prevalence(self, term: str, context: str):
        raise NotImplementedError("subclass should implement this function")

    def calculate_confidence(self, prediction: float, bin_errors: np.array):
        y = np.digitize(prediction, np.linspace(0, 1, bin_errors.shape[0] + 1))
        return 1 - bin_errors[y - 1]
