#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import List, Optional
from dotenv import dotenv_values
import openai

import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from flair.data import Sentence
from flair.models import SequenceTagger
from pydantic import BaseModel

app = FastAPI(title="simplerad API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="http://localhost:.*",
    allow_methods=["GET", "POST"],
)
nltk_tokenizer = nltk.data.load("tokenizers/punkt/dutch.pickle")
entity_taggers = {
    "ner-english": SequenceTagger.load("flair/ner-english"),
    "ner-english-fast": SequenceTagger.load("flair/ner-english-fast"),
}

openai.api_key = dotenv_values(".env")["OPENAI_API_KEY"]


class TextRequest(BaseModel):
    text: str
    model_name: str


class EntityResponse(BaseModel):
    class Span(BaseModel):
        start: int
        end: int
        text: str
        label: Optional[str]

    text: str
    sentences: List[Span]
    spans: List[Span]


@app.get("/")
def status():
    return ""


@app.get("/settings")
def settings():
    return {
        "entities": [
            {
                "label": "engine",
                "type": "select",
                "default": "ner-english",
                "values": list(entity_taggers.keys()),
            }
        ],
        "summarize": [
            {"label": "engine", "type": "select", "values": ["openai", "custom"]},
            {"label": "prompt prefix", "type": "text"},
        ],
    }


def preprocess(text: str):
    normalized = re.sub(r"\s+", " ", text)
    sents = nltk_tokenizer.tokenize(normalized, realign_boundaries=True)
    # calculate boundaries...
    bounds = [(0, len(sents[0]))]
    for i, sent in enumerate(sents[1:], start=1):
        start = bounds[i - 1][1] + 1  # account for whitespace between sentences
        bounds.append((start, start + len(sent)))
    return {
        "text": normalized,
        "sentences": [
            {"start": start, "end": end, "text": sent}
            for ((start, end), sent) in zip(bounds, sents)
        ],
    }


@app.post("/entities/", response_model=List[EntityResponse])
def get_entities(req: List[TextRequest]):
    result = []
    for r in req:
        preprocessed = preprocess(r.text)
        tmp = Sentence(preprocessed["text"])
        entity_taggers[r.model_name].predict(tmp)

        preprocessed["spans"] = [
            {
                "start": x.start_pos,
                "end": x.end_pos,
                "label": x.tag,
                "text": x.text,
            }
            for x in tmp.get_spans("ner")
        ]
        result.append(preprocessed)
    return result


@app.post("/search/", response_model=List[str])
def get_entities_detail(req: List[TextRequest]):
    return [f"Detailed information about {r.text}" for r in req]


@app.post(
    "/summarize/",
)
def summarize(req: List[TextRequest]):
    result = []
    for r in req:
        preprocessed_text = preprocess(r.text)
        if r.model_name == "openai":
            result.append(
                openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=f"Explain this in Dutch like I am 12: {preprocessed_text}",
                    max_tokens=2048,
                    temperature=0.9,
                    n=5,
                )
            )
        elif r.model_name == "custom":
            result.append({"summary": r.text})
    return result
