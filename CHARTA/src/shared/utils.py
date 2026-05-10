import json, logging
from pathlib import Path

def load_json(path: str) -> dict:
    """Load JSON file and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: dict, path: str) -> None:
    """Save dict to JSON with UTF-8 and pretty indentation."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_logger(name: str) -> logging.Logger:
    """Return logger writing to console (INFO) and logs/charta.log (DEBUG)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
        fh = logging.FileHandler("logs/charta.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
        logger.addHandler(ch); logger.addHandler(fh)
    return logger
