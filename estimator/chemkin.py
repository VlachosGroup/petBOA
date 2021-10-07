"""
Method to call openmkm's executable to
run a micro-kinetic model.
"""
import os
import shutil
import subprocess
from pathlib import Path


def clean_folder(*args):
    for ext in args:
        for name in Path(".").glob(ext):
            os.remove(name)


class ChemKIN:
    """ A simple omkm executable handler"""
    kind = "mkm executable"

    def __init__(self,
                 exe_path,
                 wd_path,
                 save_folders=True,
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
        self.process_instance = None
        self.clean = clean_folder
        self.run_number = 0

    def run(self):
        os.chdir(self.wd_path)
        if self.clean:
            os.remove("OUT.d")
            clean_folder("*.csv","*.out")
        self.process_instance = subprocess.run(args=[self.exe_path, self.reactor_file, self.thermo_file],
                                               capture_output=True, text=True,
                                      )
        print("MKM Run {} Finished".format(self.run_number))
        self.run_number += 1
        if self.save_folders:
            self.clone_folder(run_no=self.run_number)

    def clone_folder(self, run_no):
        src = self.wd_path
        dest = self.wd_path + "_" + str(run_no)  # temp naming convention
        if os.path.exists(dest): shutil.rmtree(dest)
        shutil.copytree(src, dest)
