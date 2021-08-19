import os
import time
from estimator.omkm import OMKM


def main():
    omkm_path = "C:\\Users\\skasiraj\\source\\repos\\openmkm-VS\\x64\\Debug\\openmkm-VS.exe"
    omkm_instance = OMKM(exe_path=omkm_path,
                         wd_path=os.getcwd(),
                         save_folders=False,
                         )
    omkm_instance.run()
    omkm_instance.run()
    omkm_instance.run()
    omkm_instance.run()
    omkm_instance.run()
    # out = omkm_instance.process_instance.stdout
    # err = omkm_instance.process_instance.stderr
    # returncode = stdout = omkm_instance.process_instance.returncode
    # args = omkm_instance.process_instance.args
    # print("Results")
    # print(returncode, args)


if __name__ == "__main__":
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
