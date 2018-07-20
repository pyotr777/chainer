#!/usr/bin/env python

# Prepares and runs multiple tasks on multiple GPUs: one task per GPU.
# Waits if no GPUs available. For GPU availability check uses "nvidia-smi dmon" command.

# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

import subprocess
import re
import time
import os
import numpy as np
import random
import multigpuexec


# Runs a task on specified GPU
def runProfilingTask(task,gpu):
    command = task["comm"]+" -g "+str(gpu)
    # IMPORTANT: remote double spaces or they will become empty arguments!
    command = re.sub('\s+',' ',command).strip()

    # Use unique file names for log files
    filename_base = os.path.splitext(task["logfile"])[0]
    trace_file = filename_base+"_profiling.log"
    fsmi = open(trace_file,"ab")
    # Save command into shell script file
    command_file = open(filename_base+"_com.sh","ab")
    print "command:",command,"saved to",command_file
    command_file.write(command)
    command_file.close()
    chmod="chmod +x "+str(command_file)
    RC=subprocess.call(chmod.split(" "))
    if RC > 0:
        print "Error accessing {}: {}".format(command_file,RC)
        return

    # Start profiliers
    # Command for profiling is ./<command_file>
    command1 = "./"+command_file
    prof_command = "./comb_profile.sh {command} {filename_base} &".format(command=command1,filename_base=filename_base)
    prof_pid=subprocess.Popen(prof_command.split(" "),stdout=fsmi,stderr=fsmi,bufsize=1).pid
    print "Combined profilier started with PID",prof_pid



gpus = range(0,8)
print gpus
runs = 1
tasks = []
logdir="logs/combined_profiles/"
if not os.path.exists(logdir):
    os.makedirs(logdir)
print("Logs are in",logdir)
batchsizes = random.sample(np.arange(64,900),10)
learnrates=[0.15]
epochs=1
for run in range(runs):
    for batch in batchsizes:
        for lr in learnrates:
            logfile=os.path.join(logdir,"cifar_log_b{}_l{:.3f}_{:02d}.log".format(batch,lr,run))
            if os.path.isfile(logfile):
                print "file",logfile,"exists."
                continue

            task = {"comm":"python chainer/examples/cifar/train_cifar.py -d cifar100 -e {epochs} -b {bs} -l {lr}".format(epochs=epochs,bs=batch,lr=lr,run=run),"logfile":logfile,"batch":batch,"lr":lr}
            tasks.append(task)

print "Have",len(tasks),"tasks"
gpu = -1
for i in range(0,len(tasks)):
    #print "Preapare",tasks[i]["comm"],">",tasks[i]["logfile"]
    gpu = multigpuexec.getNextFreeGPU(gpus, start=gpu+1)
    gpu_info = multigpuexec.getGPUinfo(gpu)
    f = open(tasks[i]["logfile"],"w+")
    f.write("b{} l{}\n".format(tasks[i]["batch"],tasks[i]["lr"]))
    f.write("GPU: {}\n".format(gpu_info))
    f.close()
    runProfilingTask(tasks[i],gpu)
    time.sleep(15)


