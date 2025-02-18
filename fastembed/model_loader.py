from pathlib import Path
import json
from typing import Dict, List


class ModelLoader:
    def __init__(self):
        self.config_dir = Path(__file__).parent / "configs"
        self._models: Dict[str, List[Dict]] = {}

    def load_models(self, model_type: str) -> List[Dict]:
        if model_type not in self._models:
            config_path = self.config_dir / f"{model_type}_models.json"
            with open(config_path) as f:
                self._models[model_type] = json.load(f)["models"]
        return self._models[model_type]
