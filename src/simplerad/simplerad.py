#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from time import perf_counter
from typing import List

import hydra
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import DictConfig

from .entities import entity_taggers, get_entities
from .frequency import frequencizers, get_frequencies
from .schemas import (
    EntityTaggerResponse,
    FrequencyResponse,
    SearchResponse,
    SummaryResponse,
    TextRequest,
)
from .search import get_search_results, searchers
from .summarization import get_summaries, summarizers

logger = logging.getLogger("uvicorn")

app = FastAPI(title="simplerad API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="http://localhost:.*",
    allow_methods=["GET", "POST", "PUT"],
    expose_headers=["X-Process-Time"],
)

model_dicts = {
    "entities": entity_taggers,
    "frequency": frequencizers,
    "search": searchers,
    "summarize": summarizers,
}


@app.middleware("http")
async def processing_time_logger(request, call_next):
    start_time = perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{perf_counter() - start_time:.3f}s"
    return response


@app.get("/")
def status():
    return ""


@app.post("/entities/", response_model=List[EntityTaggerResponse])
def entities(req: List[TextRequest]):
    logger.info(f"> entities - processing {len(req)} items")
    return [get_entities(r.text) for r in req]


@app.post("/search/", response_model=List[SearchResponse])
def search(req: List[TextRequest]):
    logger.info(f"> search - processing {len(req)} items")
    return [get_search_results(r.text) for r in req]


@app.post("/summarize/", response_model=List[SummaryResponse])
def summarize(req: List[TextRequest]):
    logger.info(f"> summarize - processing {len(req)} items")
    return [get_summaries(r.text) for r in req]


@app.post("/frequency/", response_model=List[FrequencyResponse])
def frequency(req: List[TextRequest]):
    logger.info(f"> frequency - processing {len(req)} items")
    return [get_frequencies(r.text) for r in req]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # set the configuration built with Hydra
    for module, models in model_dicts.items():
        models.set_config(cfg[module])

    uvicorn.run(app)


if __name__ == "__main__":
    main()
