from flair.data import Sentence
from flair.models import SequenceTagger
from hydra import compose
from omegaconf import DictConfig

from .base import BasePredictor


class FlairPredictor(BasePredictor):
    # small wrapper around Flair entity tagger

    def __init__(self, cfg: DictConfig):
        self.model = SequenceTagger.load(cfg["model_name"])

    def predict(self, text):
        s = Sentence(text["text"])
        self.model.predict(s)

        return [
            {
                "start": (start := x.start_position),
                "end": (end := x.end_position),
                "text": x.text,
            }
            for x in s.get_spans("ner")
        ]
