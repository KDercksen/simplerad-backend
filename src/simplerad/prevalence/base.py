#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class BasePrevalence:
    def get_global_prevalence(self, term: str):
        raise NotImplementedError("subclass should implement this function")

    def get_local_prevalence(self, term: str, context: str):
        raise NotImplementedError("subclass should implement this function")

    def calculate_confidence(
        self, prediction: float, bin_errors: np.array, smooth_error_window=-1
    ):
        if smooth_error_window > 0:
            bin_errors = np.convolve(
                bin_errors, np.ones(smooth_error_window) / smooth_error_window, "same"
            )
        y = np.digitize(prediction, np.linspace(0, 1, bin_errors.shape[0] + 1))
        return 1 - bin_errors[y - 1]
