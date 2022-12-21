#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from hydra import compose
from omegaconf import DictConfig

from ..utils import read_jsonl_dir
from .base import BaseSearcher


class ExactJSONLFolderSearcher(BaseSearcher):
    def __init__(self, cfg: DictConfig):
        self.known_entities = read_jsonl_dir(cfg["jsonl_directory"])

    def search(self, text: str):
        return sorted(
            [
                {"score": 1, "entity": entity}
                for entity in self.known_entities
                if text.lower() in entity["title"]
                or text.lower() in entity["description"]
                or any(text.lower() in s for s in entity.get("synonyms", []))
            ],
            key=lambda x: x["entity"]["title"],
        )
