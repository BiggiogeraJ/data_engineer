import os
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

class ModelProvider(str,Enum):
    OLLAMA = "ollama"

@dataclass
class ModelConfig:
    provider: ModelProvider
    name: str
    temperature: float

LLAMA_3_2 = ModelConfig(
    provider=ModelProvider.OLLAMA,
    name="llama3.2",
    temperature=0.1
)

class Config:
    SEED = 42
    MODEL = LLAMA_3_2
    OLLAMA_CONTEXT_WINDOW = 4096

    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATA_DIR = APP_HOME / "sqldata"
        DATABASE_PATH = DATA_DIR / "bank_database.sqlite"


def seed_everything(seed: int = Config.SEED):
    random.seed(seed)
