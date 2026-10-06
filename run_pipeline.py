"""Run the whole Person 1 pipeline in order.

    python run_pipeline.py              # everything after the (slow) download + filtering
    python run_pipeline.py --from-raw   # also download the OFF export and filter it again (~10 min, 1.3 GB)

Steps (each is also runnable on its own with `python -m <module>`):
  [raw]  src.data.download_dataset      download the Open Food Facts export
  [raw]  src.data.filter_dataset        -> data/processed/products.csv
         src.data.download_taxonomy     -> additives_reference.csv, function_classes.csv
         src.data.eda                   -> reports/figures/01-06, dataset_statistics.md
         src.labeling.build_vocabulary  -> ingredient_vocabulary.txt
         src.labeling.build_silver      -> data/processed/silver.jsonl.gz
         src.labeling.silver_report     -> reports/figures/07-08, silver_statistics.md
         src.data.split_dataset         -> data/splits/...
         src.data.export_processed      -> ingredients_processed.csv, silver_entities.csv.gz
         src.annotation.sample_gold     -> data/annotations/batches/...
         src.evaluation.off_agreement   -> reports/off_additive_agreement.json
         src.evaluation.noise_robustness-> reports/noise_robustness.json, figure 11
  [gold] src.annotation.import_annotations, src.evaluation.evaluate_baseline, src.evaluation.error_analysis
         run automatically once data/annotations/exports/ contains Doccano exports.
"""
import argparse
import importlib
import time

from src.utils.config import project_path

RAW_STEPS = ["src.data.download_dataset", "src.data.filter_dataset"]
STEPS = ["src.data.download_taxonomy", "src.data.eda", "src.labeling.build_vocabulary",
         "src.labeling.build_silver", "src.labeling.silver_report", "src.data.split_dataset",
         "src.data.export_processed", "src.annotation.sample_gold", "src.evaluation.off_agreement",
         "src.evaluation.noise_robustness"]
GOLD_STEPS = ["src.annotation.import_annotations", "src.data.split_dataset",
              "src.evaluation.evaluate_baseline", "src.evaluation.error_analysis"]


def run(module_name: str):
    print(f"\n=== {module_name} ===", flush=True)
    start = time.time()
    module = importlib.import_module(module_name)
    module.main()
    print(f"--- done in {time.time() - start:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-raw", action="store_true", help="also download and filter the raw export")
    args = parser.parse_args()

    steps = (RAW_STEPS if args.from_raw else []) + STEPS
    if list(project_path("data/annotations/exports").glob("*.jsonl")):
        steps += GOLD_STEPS
    else:
        print("No gold annotations yet (data/annotations/exports/ is empty): "
              "baseline evaluation and error analysis are skipped.")
    for step in steps:
        run(step)


if __name__ == "__main__":
    main()
