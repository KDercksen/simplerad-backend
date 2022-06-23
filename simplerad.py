#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from entities import entity_taggers, get_entities
from schemas import EntityTaggerResponse, SearchResponse, SummaryResponse, TextRequest
from search import get_search_results, searchers
from summarization import get_summary

app = FastAPI(title="simplerad API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="http://localhost:.*",
    allow_methods=["GET", "POST"],
)


@app.get("/")
def status():
    return ""


@app.get("/settings")
def settings():
    return {
        "entities": {
            "engine": {
                "default": "simstring",
                "values": list(entity_taggers.keys()),
            }
        },
        "summarize": {
            "engine": {"default": "custom", "values": ["custom"]},
            "prompt": {"default": "summarize: "},
        },
        "search": {
            "engine": {"values": list(searchers.keys()), "default": "simstring"}
        },
    }


@app.post("/entities/", response_model=List[EntityTaggerResponse])
def entities(req: List[TextRequest]):
    return [get_entities(r.text, r.model_name) for r in req]


@app.post("/search/", response_model=List[SearchResponse])
def search(req: List[TextRequest]):
    return [get_search_results(r.text, r.model_name) for r in req]


@app.post("/summarize/", response_model=List[SummaryResponse])
def summarize(req: List[TextRequest]):
    return [get_summary(r.text, r.model_name) for r in req]
