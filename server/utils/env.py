import os
import re
from dotenv import load_dotenv


def ensure_env_loaded(env_path: str = None):
    # First try python-dotenv
    try:
        load_dotenv(env_path or os.path.join(os.getcwd(), ".env"))
    except Exception:
        pass

    # If key is missing, attempt a tolerant parse of .env lines like 'KEY: "value"' or 'KEY: value'
    def set_from_file(path: str):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # match KEY: "value" or KEY: 'value' or KEY: value or KEY = value
                m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(?:\"([^\"]*)\"|'([^']*)'|([^#]*))", line)
                if not m:
                    continue
                key = m.group(1)
                val = m.group(2) or m.group(3) or m.group(4) or ""
                val = val.strip()
                if val.endswith('"') or val.endswith("'"):
                    val = val[:-1]
                if key not in os.environ:
                    os.environ[key] = val

    try:
        set_from_file(env_path or os.path.join(os.getcwd(), ".env"))
    except Exception:
        pass
