import yaml, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
def load_config(p=ROOT / "configs" / "default.yaml"):
    return yaml.safe_load(open(p))
