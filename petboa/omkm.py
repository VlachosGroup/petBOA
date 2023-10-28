"""
Method to call openmkm's executable to
run a micro-kinetic model.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path


def clean_folder(*args):
    for ext in args:
        for name in Path(".").glob(ext):
            os.remove(name)


class OMKM:
    """ A simple omkm executable handler"""
    kind = "mkm executable"

    def __init__(self,
                 exe_path=None,
                 wd_path=None,
                 save_folders=False,
                 clean_folder=True,
                 run_args=("reactor.yaml", "thermo.xml"),
                 verbose=True,
                 docker=None,
                 **kwargs):
        self.stderr = None
        self.stdout = None
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
        self.verbose = verbose
        self.docker = docker
        self.docker_exec_code = None
        self.warn_about_omkm()

    def warn_about_omkm(self):
        if self.exe_path is not None and self.docker is not None:
            raise RuntimeError("Both OpenMKM executable and Docker binary "
                               "are specified simulataneosly. \nPlease only "
                               "use either the direct executable or Docker "
                               "container.")
        if self.exe_path is None and self.docker is None:
            raise RuntimeError("Neither OpenMKM executable nor Docker binary "
                               "are specified. \nPlease specify "
                               "either the direct executable or Docker "
                               "container.")

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

        if self.docker is None:
            if not os.path.isfile(self.reactor_file):
                raise RuntimeError("Reactor file {} doesn't exist in this folder {}".format(self.reactor_file,
                                                                                            os.getcwd()))
            if not os.path.isfile(self.thermo_file):
                raise RuntimeError("Thermo file {} doesn't exist in this folder {}".format(self.thermo_file,
                                                                                           os.getcwd()))
            tic = time.perf_counter()

            if os.path.isfile(self.exe_path):
                self.process_instance = subprocess.run(args=[self.exe_path, self.reactor_file, self.thermo_file],
                                                       capture_output=True, text=True,
                                                       )
                toc = time.perf_counter()
                if self.verbose:
                    print("MKM Run {} Finished in {} seconds".format(self.run_number, toc - tic))
                self.run_number += 1
                self.stdout = self.process_instance.stdout
                self.stderr = self.process_instance.stderr
            else:
                raise RuntimeError("Either the path: {} \n"
                                   "is invalid "
                                   "or omkm executable "
                                   "doesn't exist in this path \n"
                                   "Please make sure the correct path "
                                   "to the `omkm` executable is specified".format(self.exe_path))
        if self.docker is not None:
            if self.save_folders:
                d = "run_" + str(exp_no)
            else:
                d = "run"
            working_dir = "/data/{}".format(d)  # Replace with the desired working directory
            command = ['omkm', self.reactor_file, self.thermo_file]
            tic = time.perf_counter()
            container = self.docker
            exec_command = container.exec_run(command, tty=True,
                                              workdir=working_dir)
            self.docker_exec_code = exec_command.exit_code
            self.stdout = exec_command.output
            self.stderr = exec_command.output
            toc = time.perf_counter()
            if self.verbose:
                print("MKM Run {} Finished in {} seconds".format(self.run_number, toc - tic))
            self.run_number += 1

    def clone_folder(self, run_no):
        src = self.wd_path
        dest = self.wd_path + "_" + str(run_no)  # temp naming convention
        if os.path.exists(dest): shutil.rmtree(dest)
        shutil.copytree(src, dest)
