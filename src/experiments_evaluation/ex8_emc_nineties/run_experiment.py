import subprocess

PERCENTAGES = ["92", "94", "96", "98"]


if __name__ == '__main__':
    """
    nohup python run_experiment.py > log.txt &
    """
    # Run the other script
    for percentage in PERCENTAGES:
        print(f"[*] Going to execute script using {percentage}%, using pretrained model on ex5.1-90%")
        subprocess.run(["python", "train.py",
                        "-p", percentage])
        print(f"[*] completed run-{percentage}%")
