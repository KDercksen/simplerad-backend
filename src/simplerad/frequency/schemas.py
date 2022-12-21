#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pydantic import BaseModel


class FrequencyResponse(BaseModel):
    global_frequency: float
    global_certainty: float
    local_frequency: float
    local_certainty: float
