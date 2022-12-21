#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as t
from hydra import compose
from omegaconf import DictConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base import BaseSummarizer


class TransformerAbstractiveSummarizer(BaseSummarizer):
    def __init__(self, cfg: DictConfig):
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
