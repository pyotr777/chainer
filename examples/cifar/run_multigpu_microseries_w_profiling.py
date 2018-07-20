#!/usr/bin/env python

# Prepares and runs multiple tasks on multiple GPUs: one task per GPU.
# Waits if no GPUs available. For GPU availability check uses "nvidia-smi dmon" command.

# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

import subprocess
import re
import time
import os
# impoer sys
# import numpy as np
# import random
import multigpuexec


# Runs a task on specified GPU
# Executing command :
# nvprof --csv --logfile <nvproflog> --metrics <metrics> <command>
# command:
# python chainer/examples/cifar/train... -d cifar100 ...
def runProfilingTask(task,gpu):
    #metrics = 'flop_dp_efficiency,flop_sp_efficiency,achieved_occupancy,eligible_warps_per_cycle,flop_count_dp,flop_count_sp,inst_per_warp,ipc,issue_slot_utilization,issue_slots,issued_ipc'
    metrics = 'all'

    command = task["comm"]+" -g "+str(gpu)
    # IMPORTANT: remote double spaces or they will become empty arguments!
    command = re.sub('\s+',' ',command).strip()

    # Use unique file names for log files
    filename_base = os.path.splitext(task["logfile"])[0]
    log_file = task["logfile"]
    log_fd = open(log_file,"ab")

    nvprof_log = filename_base+"_nvprof.csv"

    nvprof_com = "nvprof --csv --log-file {logfile} --metrics {metrics} {command}".format(logfile=nvprof_log, metrics=metrics,command=command)
    log_fd.write("Command={}".format(nvprof_com))
    # Start profiliers
    prof_pid=subprocess.Popen(nvprof_com.split(" "),stdout=log_fd,stderr=log_fd,bufsize=1).pid
    print "nvprof started with PID",prof_pid


gpus = range(2,8)
tasks = []
logdir="logs/nvprofiles/all_metrics/"
if not os.path.exists(logdir):
    os.makedirs(logdir)

batchsizes = range(12,1025,92)
epochs=1
iterations=2
model = "model64"
for batch in batchsizes:
    logfile=os.path.join(logdir,"cifar_mod64_log_b{}_iter{}.log".format(batch,iterations))
    if os.path.isfile(logfile):
        print "file",logfile,"exists."
        continue

    task = {"comm":"python chainer/examples/cifar/train_cifar_model.py -d cifar100 -e {epochs} -b {bs} --iterations {iter} --model {model}".format(epochs=epochs,bs=batch,iter=iterations,model=model),"logfile":logfile,"batch":batch}
    tasks.append(task)

print "Have",len(tasks),"tasks"
gpu = -1
for i in range(0,len(tasks)):
    #print "Preapare",tasks[i]["comm"],">",tasks[i]["logfile"]
    gpu = multigpuexec.getNextFreeGPU(gpus, start=gpu+1,c=20,d=1,debug=False)
    gpu_info = multigpuexec.getGPUinfo(gpu)
    f = open(tasks[i]["logfile"],"w+")
    f.write("b{}\n".format(tasks[i]["batch"]))
    f.write("GPU: {}\n".format(gpu_info))
    f.close()
    print "{}/{} tasks".format(i+1,len(tasks))
    runProfilingTask(tasks[i],gpu)
    time.sleep(15)


