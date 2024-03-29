#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from time import perf_counter
from typing import List

import hydra
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from omegaconf import DictConfig

from .classification import (
    get_text_classification,
    get_sentence_classification,
    text_classifiers,
    sentence_classifiers,
)
from .entities import entity_taggers, get_entities
from .prevalence import prevalencers, get_global_prevalence, get_local_prevalence
from .schemas import (
    EntityTaggerResponse,
    PrevalenceResponse,
    SearchResponse,
    SummaryResponse,
    TextRequest,
    TextContextRequest,
    TextClassificationResponse,
    SentenceClassificationResponse,
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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

model_dicts = {
    "entities": entity_taggers,
    "prevalence": prevalencers,
    "search": searchers,
    "summarize": summarizers,
    "text_classification": text_classifiers,
    "sentence_classification": sentence_classifiers,
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


@app.post("/prevalence/global", response_model=List[PrevalenceResponse])
def prevalence(req: List[TextRequest]):
    logger.info(f"> prevalence/global - processing {len(req)} items")
    return [get_global_prevalence(r.text) for r in req]


@app.post("/prevalence/local", response_model=List[PrevalenceResponse])
def prevalence(req: List[TextContextRequest]):
    logger.info(f"> prevalence/local - processing {len(req)} items")
    return [get_local_prevalence(r.text, r.context) for r in req]


@app.post(
    "/sentence_classification/", response_model=List[SentenceClassificationResponse]
)
def sentence_classification(req: List[TextRequest]):
    logger.info(f"> sentence classification - processing {len(req)} items")
    return [get_sentence_classification(r.text) for r in req]


@app.post("/text_classification/", response_model=List[TextClassificationResponse])
def text_classification(req: List[TextRequest]):
    logger.info(f"> text classification - processing {len(req)} items")
    return [get_text_classification(r.text) for r in req]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # set the configuration built with Hydra
    for module, models in model_dicts.items():
        models.set_config(cfg[module])

    uvicorn.run(app)


if __name__ == "__main__":
    main()
