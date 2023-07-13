#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from omegaconf import DictConfig
from .base import BasePrevalence
import torch as t


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

    def get_global_prevalence(self, term: str):
        global_inputs = self.tokenizer(term, return_tensors="pt").to(self.device)
        with t.no_grad():
            set_peft_model_state_dict(self.model, self.global_state_dict)
            global_prevalence = (
                t.sigmoid(self.model(**global_inputs).logits).cpu().numpy()[0]
            )

            # TODO:
            global_certainty = 1

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

            # TODO:
            local_certainty = 1

        return local_prevalence, local_certainty
