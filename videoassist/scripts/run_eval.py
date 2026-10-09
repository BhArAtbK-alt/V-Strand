"""python scripts/run_eval.py  -> prints + saves ablation table (outputs/eval_results.{json,md})"""
import sys, json, glob, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from videoassist.config import load_config
from videoassist.evaluate import evaluate, to_markdown
videos = [json.load(open(f)) for f in sorted(glob.glob(str(ROOT / "data/staged/*.json")))]
assert videos, "no data: run `python scripts/make_dataset.py` (or add your annotated staged videos)"
res = evaluate(videos, load_config())
md = to_markdown(res); print(f"{len(videos)} videos\n" + md)
(ROOT / "outputs" / "eval_results.json").write_text(json.dumps(res, indent=1)); (ROOT / "outputs" / "eval_results.md").write_text(md)
