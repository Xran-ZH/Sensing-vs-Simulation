import argparse

from src import PLUS_STATE, make_config, run_line_scan


DATA_SUBDIR = "line"
EXPERIMENT_NAME = "line"

N = 10
TIME = 1.0
INITIAL_STATE = PLUS_STATE

SCAN = {
    "type": "line",
    "description": "delta = epsilon * direction",
    "direction": [1.0, 0.0],
    "epsilon": {"start": -0.3, "stop": 0.3, "num": 11},
}


def build_config(smoke=False):
    n = 3 if smoke else N
    scan = {
        **SCAN,
        "epsilon": {"start": -0.05, "stop": 0.05, "num": 3} if smoke else SCAN["epsilon"],
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

    output_dir = run_line_scan(build_config(smoke=args.smoke))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
