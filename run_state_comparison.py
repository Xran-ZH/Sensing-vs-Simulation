import argparse

from experiment_utils import (
    CHAOTIC_EVOLVED_STATE,
    GHZ_STATE,
    PLUS_STATE,
    RANDOM_PRODUCT_STATE,
    make_config,
    run_line_scan,
)


DATA_SUBDIR = "state_comparison"
EXPERIMENT_NAME = "state_comparison"

N = 10
TIME = 1.0

SCAN = {
    "type": "line",
    "description": "delta = epsilon * direction",
    "direction": [1.0, 0.0],
    "epsilon": {"start": -0.3, "stop": 0.3, "num": 9},
}

STATES = {
    "plus": PLUS_STATE,
    "GHZ": GHZ_STATE,
    "random product": RANDOM_PRODUCT_STATE,
    "chaotic evolved": CHAOTIC_EVOLVED_STATE,
}


def build_configs(smoke=False):
    n = 3 if smoke else N
    scan = {
        **SCAN,
        "epsilon": {"start": -0.05, "stop": 0.05, "num": 3} if smoke else SCAN["epsilon"],
    }
    for label, initial_state in STATES.items():
        yield make_config(
            experiment_name=f"{EXPERIMENT_NAME}_{label}",
            data_subdir=DATA_SUBDIR,
            n=n,
            time=TIME,
            initial_state=initial_state,
            scan=scan,
            smoke=smoke,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run tiny sanity-check datasets.")
    args = parser.parse_args()

    for config in build_configs(smoke=args.smoke):
        output_dir = run_line_scan(config)
        print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
