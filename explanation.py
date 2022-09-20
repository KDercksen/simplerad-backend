#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shelve
from collections import defaultdict
from pathlib import Path

import openai
import spacy
from dotenv import dotenv_values
from hydra import compose

from utils import LazyValueDict, lemmatize_tokens, preprocess


class BaseExplainer:
    def explain(self, text: str):
        raise NotImplementedError("subclass should implement this function")

    def propose(self, text: str):
        raise NotImplementedError("subclass should implement this function")

    def add_to_cache(self, key: str, text: str):
        raise NotImplementedError("subclass should implement this function")


class GPT3CacheExplainer(BaseExplainer):
    def __init__(self):
        cfg = compose(config_name="config", overrides=["explanation=gpt3"])[
            "explanation"
        ]
        openai.api_key = dotenv_values(".env")["OPENAI_API_KEY"]
        self.model = cfg["model"]
        self.nlp = spacy.load(cfg["spacy_model"])
        self.preprompt = cfg["preprompt"]
        self.temperature = cfg["temperature"]
        self.max_tokens = cfg["max_tokens"]
        self.shelve_path = cfg["shelve_path"]

        # initialize if necessary with a defaultdict, makes the rest of cache handling
        # easier
        if not Path(self.shelve_path).exists():
            with shelve.open(self.shelve_path) as db:
                db["dict"] = defaultdict(list)

    def explain(self, text: str):
        with shelve.open(self.shelve_path) as db:
            return db["dict"][lemmatize_tokens(text, self.nlp)]  # type: ignore

    def propose(self, text: str):
        # make prompt:
        prompt = self.preprompt + f'"{text}"'
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return [response["choices"][0]["text"].strip()]

    def add_to_cache(self, key: str, text: str):
        with shelve.open(self.shelve_path) as db:
            tmp = db["dict"]
            tmp[lemmatize_tokens(key, self.nlp)].append(text)  # type: ignore
            db["dict"] = tmp


explainers = LazyValueDict(
    {
        "gpt3": GPT3CacheExplainer,
    }
)


def get_explanations(text: str, model_name: str):
    preprocessed = preprocess(text)
    results = explainers[model_name].explain(preprocessed["text"])
    return {"data": results}


def propose_explanation(text: str, model_name: str):
    preprocessed = preprocess(text)
    results = explainers[model_name].propose(preprocessed["text"])
    return {"data": results}


def add_explanation(text: str, model_name: str, term: str):
    preprocessed = preprocess(term)  # key
    # text is description
    explainers[model_name].add_to_cache(preprocessed["text"], text)  # key, value
