import yaml
from pathlib import Path
from functools import lru_cache

PROMPTS_FILE = Path(__file__).parent / "system_prompts.yaml"

@lru_cache
def load_prompts():
    """Carga los prompts desde el archivo YAML."""
    if not PROMPTS_FILE.exists():
        return {}
        
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data.get("system_prompts", {})

def get_prompt(key: str) -> str:
    """Obtiene un prompt específico por su clave."""
    prompts = load_prompts()
    return prompts.get(key, "")
