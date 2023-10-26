"""
Method to call openmkm's executable to
run a micro-kinetic model.
"""
import glob
import os
import time
import shutil
import subprocess
from pathlib import Path


def clean_folder(*args):
    for ext in args:
        for name in Path(".").glob(ext):
            os.remove(name)


class OMKM:
    """ A simple omkm executable handler"""
    kind = "mkm executable"

    def __init__(self,
                 exe_path,
                 wd_path,
                 save_folders=False,
                 clean_folder=True,
                 run_args=("reactor.yaml", "thermo.xml"),
                 **kwargs):
        self.exe_path = exe_path
        self.wd_path = wd_path
        self.save_folders = save_folders
        if "slurm_required" in kwargs.keys():
            self.slurm_required = kwargs["slurm_required"]
            self.slurm_file_name = kwargs["slurm_file"]
        else:
            self.slurm_required = False
            self.slurm_file_name = None
        self.reactor_file = run_args[0]
        self.thermo_file = run_args[1]
        self.process_instance = None
        self.clean = clean_folder
        self.run_number = 0

    def run(self, exp_no):
        os.chdir(self.wd_path)
        if self.save_folders:
            if not os.path.exists("run_" + str(exp_no)):
                os.mkdir("run_" + str(exp_no))
            os.chdir("run_" + str(exp_no))
        else:
            if not os.path.exists("run"):
                os.mkdir("run")
            os.chdir("run")
        if self.clean:
            clean_folder("*.csv", "*.out")
        shutil.copy(os.path.join(self.wd_path, self.reactor_file), ".")
        shutil.copy(os.path.join(self.wd_path, self.thermo_file), ".")
        tic = time.perf_counter()
        self.process_instance = subprocess.run(args=[self.exe_path, self.reactor_file, self.thermo_file],
                                               capture_output=True, text=True,
                                               )
        toc = time.perf_counter()
        print("MKM Run {} Finished in {} seconds".format(self.run_number, toc - tic))
        self.run_number += 1

    def clone_folder(self, run_no):
        src = self.wd_path
        dest = self.wd_path + "_" + str(run_no)  # temp naming convention
        if os.path.exists(dest): shutil.rmtree(dest)
        shutil.copytree(src, dest)
