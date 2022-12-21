#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from pydantic import BaseModel


class SearchResponse(BaseModel):
    class SearchResult(BaseModel):
        class Entity(BaseModel):
            title: str
            description: str
            url: str
            source_id: str
            source: str

        entity: Entity
        score: float

    data: List[SearchResult]
