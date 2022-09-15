#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
from utils import preprocess
from hydra import compose
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import LazyValueDict
import torch as t


class BaseSummarizer:
    def summarize(self, text: str):
        raise NotImplementedError("subclass should implement this function")


class TransformerAbstractiveSummarizer(BaseSummarizer):
    def __init__(self):
        cfg = compose(
            config_name="config", overrides=["summarize=transformer_abstractive"]
        )["summarize"]
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"]).to(self.device)
        self.max_len = cfg["max_generation_length"]

    def summarize(self, text: str):
        tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = (
            self.model.generate(**tokenized, max_length=self.max_len).detach().cpu()
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


class TransformerExtractiveSummarizer(BaseSummarizer):
    def __init__(self):
        cfg = compose(
            config_name="config", overrides=["summarize=transformer_extractive"]
        )["summarize"]
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"]).to(self.device)

        self.resize_position_embeddings(cfg["resize_position_embeddings"])
        self.replace_classifier(cfg["num_tf_layers"], cfg["num_tf_heads"])

    def resize_position_embeddings(self, size: int):
        hidden_size = self.model.config.hidden_size
        pos_embeddings = t.nn.Embedding(size, hidden_size)
        pos_embeddings.weight = t.nn.Parameter(
            self.model.embeddings.position_embeddings.weight.repeat(
                (size // hidden_size) + 1, 1
            )[:size]
        )
        token_type_ids = t.atleast_2d(t.zeros(size, dtype=int))
        self.model.embeddings.position_embeddings = pos_embeddings
        self.model.embeddings.token_type_ids = token_type_ids
        self.model.embeddings.position_ids = t.atleast_2d(t.arange(0, size, dtype=int))

    def replace_classifier(self, num_tf_layers: int, num_tf_heads: int):
        hidden_size = self.model.config.hidden_size
        layers = OrderedDict()
        for i in range(num_tf_layers):
            layers[f"tf_{i}"] = t.nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_tf_heads, batch_first=True
            )
        layers["linear"] = t.nn.Linear(hidden_size, 2)
        self.model.classifier = t.nn.Sequential(layers)


summarizers = LazyValueDict(
    {
        "transformer_abstractive": TransformerAbstractiveSummarizer,
        # "transformer_extractive": TransformerExtractiveSummarizer,
    }
)


def get_summaries(text: str, model_name: str):
    preprocessed = preprocess(text)
    summary = summarizers[model_name].summarize(preprocessed["text"])
    return {"summary": summary}
