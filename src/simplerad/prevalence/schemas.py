#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pydantic import BaseModel


class PrevalenceResponse(BaseModel):
    prevalence: float
    certainty: float
