import argparse

from experiment_utils import PLUS_STATE, make_config, run_grid2d_scan


DATA_SUBDIR = "grid2d"
EXPERIMENT_NAME = "grid2d"

N = 10
TIME = 1.0
INITIAL_STATE = PLUS_STATE

SCAN = {
    "type": "grid2d",
    "description": "Two-dimensional scan over delta_0 and delta_1",
    "delta_0": {"start": -0.4, "stop": 0.4, "num": 40},
    "delta_1": {"start": -0.4, "stop": 0.4, "num": 40},
}


def build_config(smoke=False):
    n = 3 if smoke else N
    scan = {
        **SCAN,
        "delta_0": {"start": -0.05, "stop": 0.05, "num": 3} if smoke else SCAN["delta_0"],
        "delta_1": {"start": -0.05, "stop": 0.05, "num": 3} if smoke else SCAN["delta_1"],
    }
    return make_config(
        experiment_name=EXPERIMENT_NAME,
        data_subdir=DATA_SUBDIR,
        n=n,
        time=TIME,
        initial_state=INITIAL_STATE,
        scan=scan,
        smoke=smoke,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a tiny sanity-check dataset.")
    args = parser.parse_args()

    output_dir = run_grid2d_scan(build_config(smoke=args.smoke))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
