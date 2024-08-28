import subprocess


ACTIVATION_FUNCTION_VARIANTS = [str(v) for v in (range(1, 10))]


if __name__ == '__main__':
    """
    nohup python run_experiment.py > log.txt &
    """
    # Run the other script
    for af_variant in ACTIVATION_FUNCTION_VARIANTS:
        print(f"[*] Going to execute script using activation function variant {af_variant}%")
        subprocess.run(["python", "train.py",
                        "-a", af_variant])
        print(f"[*] completed run {af_variant}")
