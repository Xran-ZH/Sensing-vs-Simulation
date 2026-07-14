import argparse

from experiment_utils import (
    CHAOTIC_EVOLVED_STATE,
    GHZ_STATE,
    PLUS_STATE,
    RANDOM_PRODUCT_STATE,
    make_config,
    run_qfi_scaling,
)


DATA_SUBDIR = "qfi_scaling"
EXPERIMENT_NAME = "qfi_scaling"

TIME = 1.0

SCAN = {
    "type": "qfi_scaling",
    "description": "Directional QFI scaling with system size along the zfield perturbation",
    "n_values": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "direction": [0.0, 1.0],
    "states": {
        "plus": PLUS_STATE,
        "GHZ": GHZ_STATE,
        "random product": RANDOM_PRODUCT_STATE,
        "chaotic evolved": CHAOTIC_EVOLVED_STATE,
    },
}


def build_config(smoke=False):
    scan = {
        **SCAN,
        "n_values": [2, 3] if smoke else SCAN["n_values"],
    }
    return make_config(
        experiment_name=EXPERIMENT_NAME,
        data_subdir=DATA_SUBDIR,
        n=scan["n_values"][0],
        time=TIME,
        initial_state=PLUS_STATE,
        scan=scan,
        smoke=smoke,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a tiny sanity-check dataset.")
    args = parser.parse_args()

    output_dir = run_qfi_scaling(build_config(smoke=args.smoke))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
