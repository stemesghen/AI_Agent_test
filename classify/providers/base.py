from abc import ABC, abstractmethod
from typing import Dict

class Classifier(ABC):
    @abstractmethod
    def classify(self, text: str) -> Dict:
        """Return dict with keys:
        is_incident (bool), incident_types (list[str]), near_miss (bool),
        confidence (float 0..1), rationale (str <= 60 chars)
        """
        ...

