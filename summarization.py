#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import preprocess
from hydra import initialize, compose
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import LazyValueDict
import torch as t


class BaseSummarizer:
    def summarize(self, text: str):
        raise NotImplementedError("subclass should implement this function")


class TransformerSummarizer(BaseSummarizer):
    def __init__(self):
        with initialize("conf", version_base="1.1"):
            cfg = compose("summarize/transformer.yaml")["summarize"]
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


summarizers = LazyValueDict(
    {
        "transformer": TransformerSummarizer,
    }
)


def get_summary(text: str, model_name: str):
    preprocessed = preprocess(text)
    summary = summarizers[model_name].summarize(preprocessed["text"])
    return {"summary": summary}
