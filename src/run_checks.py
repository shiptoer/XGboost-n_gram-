import compileall
import importlib
from pathlib import Path
import sys
import traceback


SRC_DIR = Path(__file__).resolve().parent
MODULES = [
    "api",
    "project_utils",
    "train_xgboost",
    "retrain_with_honeypot",
    "compare_models",
    "compare_ngram",
    "setup_sample_model",
    "test_model_smoke",
    "test_api_latency",
    "test_honeypot_payloads",
    "apply_fixes",
]


def main():
    print("Running compileall...")
    ok = compileall.compile_dir(str(SRC_DIR), quiet=1)
    print("compileall_result:", ok)

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    for module_name in MODULES:
        try:
            if module_name in sys.modules:
                del sys.modules[module_name]
            importlib.import_module(module_name)
            print(f"IMPORT_OK:{module_name}")
        except Exception as e:
            print(f"IMPORT_ERROR:{module_name}:{type(e).__name__}:{e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
