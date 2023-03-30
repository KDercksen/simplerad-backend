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
    def __init__(self, cfg: DictConfig):
        sklearn_model_path = Path(cfg["sklearn_model_path"])
        self.regression_model = joblib.load(sklearn_model_path / "regression_model.pkl")
        # this is an array of mean absolute errors for each specific prediction bin
        self.bin_errors = np.load(sklearn_model_path / "bin_errors.npy")
        self.embeddings = TransformerDocumentEmbeddings(cfg["embedding_model_path"])

    def calculate_confidence(self, prediction: float):
        y = np.digitize(prediction, np.linspace(0, 1, self.bin_errors.shape[0] + 1))
        return 1 - self.bin_errors[y - 1]

    def get_prevalence(self, term: str):
        # get transformer embedding
        s = Sentence(term)
        self.embeddings.embed(s)
        e = s.embedding.cpu().detach().numpy()
        global_prevalence = self.regression_model.predict(e.reshape(1, -1))
        global_certainty = self.calculate_confidence(global_prevalence)
        return global_prevalence, global_certainty
