import argparse
import subprocess
import sys

def run_freq_model(args):
    cmd = [
        "python", "src/mlflow/trainer_freq.py",
        args.remote_server_uri,
        args.experiment_name,
        args.run_name,
        args.kernel,
        str(args.degree),
        args.class_weight
    ]
    subprocess.run(cmd, check=True)

def run_reg_model(args):
    cmd = [
        "python", "src/mlflow/trainer_reg.py",
        args.remote_server_uri,
        args.experiment_name,
        args.run_name,
        str(args.n_estimators),
        str(args.max_depth)
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run freq and/or reg models with hyperparameters.")

    parser.add_argument("--remote_server_uri", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="Optimisation")
    parser.add_argument("--run_name", type=str, default="default")

    # Flags to activate each model
    parser.add_argument("--run_freq", action="store_true", help="Run frequency model")
    parser.add_argument("--run_reg", action="store_true", help="Run regression model")

    # Freq model hyperparams
    parser.add_argument("--kernel", type=str, default="poly")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--class_weight", type=str, default="balanced")

    # Reg model hyperparams
    parser.add_argument("--n_estimators", type=int, default=4)
    parser.add_argument("--max_depth", type=int, default=7)

    args = parser.parse_args()

    if not args.run_freq and not args.run_reg:
        print("⚠️ You must specify at least one model to run: --run_freq or --run_reg")
        sys.exit(1)

    if args.run_freq:
        print("Training Frequency Model...")
        run_freq_model(args)

    if args.run_reg:
        print("Training Regression Model...")
        run_reg_model(args)

if __name__ == "__main__":
    main()
