import subprocess

PERCENTAGES = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "99"]
LOADINGS = ["", "10", "20", "30", "40", "50", "60", "70", "80", "90"]


if __name__ == '__main__':
    """
    nohup python run_experiment.py > log.txt &
    """
    # Run the other script
    for load_id, percentage in enumerate(PERCENTAGES):
        load_last = LOADINGS[load_id]
        if load_last:
            print(f"[*] Going to execute script using {percentage}%, using pretrained model on {load_last}%")
        else:
            print(f"[*] Going to execute script using {percentage}% without loading a pretrained model")

        subprocess.run(["python", "train.py",
                        "-p", percentage,
                        "-l", load_last])
        print(f"[*] completed run-{percentage}%")
