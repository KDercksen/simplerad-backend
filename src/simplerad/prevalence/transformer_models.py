#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import torch as t
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import BasePrevalence


class GlobalLocalAdapterPrevalence(BasePrevalence):
    def __init__(self, cfg: DictConfig):
        base_model_name = cfg["base_model"]
        global_adapter = Path(cfg["global_adapter"])
        local_adapter = Path(cfg["local_adapter"])

        self.device = t.device("cuda") if t.cuda.is_available() else "cpu"
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=1
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        global_lora_config = LoraConfig.from_pretrained(
            global_adapter / "adapter_model"
        )
        local_lora_config = LoraConfig.from_pretrained(local_adapter / "adapter_model")
        assert (
            global_lora_config == local_lora_config
        ), "This class only supports two adapters with the same config"
        self.model = get_peft_model(self.base_model, global_lora_config)
        self.model.to(self.device)

        self.global_state_dict = t.load(
            global_adapter / "adapter_model/adapter_model.bin"
        )
        self.local_state_dict = t.load(
            local_adapter / "adapter_model/adapter_model.bin"
        )

        self.global_bin_errors = np.load(global_adapter / "bin_errors.npy")
        self.local_bin_errors = np.load(local_adapter / "bin_errors.npy")
        # default sub-0 means no smoothing
        self.smooth_error_window = cfg.get("smooth_error_window", -1)

    def get_global_prevalence(self, term: str):
        global_inputs = self.tokenizer(term, return_tensors="pt").to(self.device)
        with t.no_grad():
            set_peft_model_state_dict(self.model, self.global_state_dict)
            global_prevalence = (
                t.sigmoid(self.model(**global_inputs).logits).cpu().numpy()[0]
            )

            global_certainty = self.calculate_confidence(
                global_prevalence,
                self.global_bin_errors,
                smooth_error_window=self.smooth_error_window,
            )

        return global_prevalence, global_certainty

    def get_local_prevalence(self, term: str, context: str):
        local_inputs = self.tokenizer(term, context, return_tensors="pt").to(
            self.device
        )
        with t.no_grad():
            set_peft_model_state_dict(self.model, self.local_state_dict)
            local_prevalence = (
                t.sigmoid(self.model(**local_inputs).logits).cpu().numpy()[0]
            )

            local_certainty = self.calculate_confidence(
                local_prevalence,
                self.local_bin_errors,
                smooth_error_window=self.smooth_error_window,
            )

        return local_prevalence, local_certainty
