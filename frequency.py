#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from annoy import AnnoyIndex
from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from hydra import compose

from utils import LazyValueDict, preprocess


class BaseFrequency:
    def get_frequency(self, term: str):
        raise NotImplementedError("subclass should implement this function")


class TransformerFrequency(BaseFrequency):
    def __init__(self):
        cfg = compose(config_name="config", overrides=["frequency=transformer"])[
            "frequency"
        ]
        index_path = cfg["index_path"]
        embedding_dim = cfg["embedding_dim"]
        metric = cfg["metric"]
        transformer_model = cfg["transformer_model"]
        self.auc_lower_y = cfg["auc_lower_y"]
        self.min_similarity = cfg["min_similarity"]
        self.num_retrieve = cfg["num_retrieve"]

        self.index = AnnoyIndex(embedding_dim, metric)
        self.index.load(index_path)
        self.embedding = TransformerDocumentEmbeddings(transformer_model)
        self.scaling_auc = np.trapz(np.full(self.num_retrieve, 1 - self.min_similarity))

    def make_transformer_embedding(self, text: str):
        s = Sentence(text)
        self.embedding.embed(s)
        e = s.embedding.cpu().detach().numpy()
        return e / (np.linalg.norm(e) + 1e-8)

    def get_frequency(self, query: str):
        emb = self.make_transformer_embedding(query)
        _, dists = self.index.get_nns_by_vector(
            emb, self.num_retrieve, include_distances=True
        )
        # (NOTE: this is specifically meant for angular distance;
        # we probably won't use anything else, but keep in mind)
        # normalize to 0-1
        dists = (2 - np.array(dists)) / 2  # type: ignore
        real_auc = np.trapz(np.clip(dists - self.auc_lower_y, 0, 1 - self.auc_lower_y))

        # 0 - 1 estimated frequency of occurrence of query depending on embedding
        # similarities
        estimated_freq = np.clip(real_auc / self.scaling_auc, 0, 1)
        # how many of retrieved top-n samples are above threshold similarity
        sample_size = (dists > self.min_similarity).sum() / self.num_retrieve

        return estimated_freq, sample_size


frequencizers = LazyValueDict(
    {
        "transformer": TransformerFrequency,
    }
)


def get_frequencies(text: str, model_name: str):
    # relative frequency based on AUC, normalized by num_samples or something
    preprocessed = preprocess(text)
    estimated_freq, sample_size = frequencizers[model_name].get_frequency(
        preprocessed["text"]
    )
    return {"estimated_frequency": estimated_freq, "sample_size": sample_size}
