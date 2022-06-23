#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utils import preprocess


def get_summary(text: str, model_name: str):
    return {"summary": preprocess(text)["text"]}
