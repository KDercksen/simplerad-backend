#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pydantic import BaseModel


class PrevalenceResponse(BaseModel):
    global_prevalence: float
    global_certainty: float
