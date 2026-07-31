import argparse

from src import (
    CHAOTIC_EVOLVED_STATE,
    DEFAULT_H0,
    GHZ_STATE,
    PLUS_STATE,
    RANDOM_PRODUCT_STATE,
    make_config,
    run_trotter_error,
)


DATA_SUBDIR = "trotter_error"
EXPERIMENT_NAME = "trotter_error"

N = 6
DELTA = 1.0
TARGET_H = DEFAULT_H0

SCAN = {
    "type": "trotter_error",
    "description": "Single-step product-formula Bures error versus the QFI of the BCH Trotter generator",
    "orders": [1, 2],
    "step_times": [0.2, 0.1, 0.05, 0.025, 0.0125],
    "delta": DELTA,
}

STATES = {
    "plus": PLUS_STATE,
    "GHZ": GHZ_STATE,
    "random product": RANDOM_PRODUCT_STATE,
    "chaotic evolved": CHAOTIC_EVOLVED_STATE,
}


def build_config(smoke=False):
    scan = {
        **SCAN,
        "step_times": [0.05, 0.025] if smoke else SCAN["step_times"],
    }
    return make_config(
        experiment_name=EXPERIMENT_NAME,
        data_subdir=DATA_SUBDIR,
        n=3 if smoke else N,
        time=None,
        initial_state=PLUS_STATE,
        scan=scan,
        smoke=smoke,
        h0=TARGET_H,
        perturbations=[],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a tiny sanity-check dataset.")
    args = parser.parse_args()

    output_dir = run_trotter_error(build_config(smoke=args.smoke), STATES)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
