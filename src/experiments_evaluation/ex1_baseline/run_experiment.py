import subprocess


PERCENTAGES = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "99"]


if __name__ == '__main__':
    """
    nohup python run_experiment.py > log.txt &
    """
    # Run the other script
    for percentage in PERCENTAGES:
        print(f"[*] Going to execute script using {percentage}%")
        subprocess.run(["python", "train.py",
                        "-p", percentage])
        print(f"[*] completed run-{percentage}%")
