import subprocess
import os


# (exp-name, run-file)
EXPERIMENTS_TO_RUN = [
    ("ex1_baseline", "run_experiment.py"),
    ("ex2_seasonal_component", "run_experiment.py"),
    ("ex3_incremental_pretraining", "run_experiment.py"),
    ("ex3.1_moving_window", "run_experiment.py"),
    ("ex3.2_cm_inclusion", "run_experiment.py"),
    ("ex3.3_elevation", "run_experiment.py"),
    ("ex3.4_pi_init", "run_experiment.py"),
    ("ex4.1_elev_mov_win", "run_experiment.py"),
    ("ex4.2_elev_cmi", "run_experiment.py"),
    ("ex4.3_elev_pi_init", "run_experiment.py"),
    ("ex5.1_elev_mov_cmi", "run_experiment.py"),
    ("ex5.2_elev_mov_pi", "run_experiment.py"),
    ("ex6.1_elev_mov_cmi_pi", "run_experiment.py"),
    ("ex7.1_emc_fine_tune", "run_experiment.py"),
    ("ex7.2_emc_mnar_training", "run_experiment.py"),
    ("ex8_emc_nineties", "run_experiment.py"),
]


if __name__ == "__main__":
    """
    nohup python run_all_experiments.py > log.txt &
    """
    orig_cwd = os.getcwd()
    for path, file_name in EXPERIMENTS_TO_RUN:
        print("==============================================")
        print("==============================================")
        print("")
        print(f"Going to run {path}/{file_name}")
        print("")
        print("==============================================")
        print("==============================================")

        # Change cwd for execution
        os.chdir(os.path.dirname(os.path.abspath(f"{path}/{file_name}")))
        # Run the process. This may take >30h !!!
        subprocess.run(["python", file_name])
        # Change cwd back
        os.chdir(orig_cwd)
