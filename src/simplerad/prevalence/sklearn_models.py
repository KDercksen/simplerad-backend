#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import joblib
import numpy as np
from omegaconf import DictConfig
from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings

from .base import BasePrevalence


class SKLearnPrevalence(BasePrevalence):
    # NOTE: this model can only predict global prevalence, and predicts 0 for local prevalence/certainty
    def __init__(self, cfg: DictConfig):
        sklearn_model_path = Path(cfg["sklearn_model_path"])
        self.regression_model = joblib.load(sklearn_model_path / "regression_model.pkl")
        # this is an array of mean absolute errors for each specific prediction bin
        self.bin_errors = np.load(sklearn_model_path / "bin_errors.npy")
        self.embeddings = TransformerDocumentEmbeddings(cfg["embedding_model_path"])
        self.smooth_error_window = cfg.get("smooth_error_window", -1)

    def get_global_prevalence(self, term: str):
        # get transformer embedding
        s = Sentence(term)
        self.embeddings.embed(s)
        e = s.embedding.cpu().detach().numpy()
        global_prevalence = np.clip(
            self.regression_model.predict(e.reshape(1, -1)), 0, 1
        )
        global_certainty = self.calculate_confidence(
            global_prevalence,
            self.bin_errors,
            smooth_error_window=self.smooth_error_window,
        )
        return global_prevalence, global_certainty

    def get_local_prevalence(self, term: str, context: str):
        print("WARNING: not implemented for this model")
        return 0, 0
