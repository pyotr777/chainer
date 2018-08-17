#!/usr/bin/env python

# Prepares and runs multiple tasks on multiple GPUs: one task per GPU.
# Waits if no GPUs available. For GPU availability check uses "nvidia-smi dmon" command.

# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

import multigpuexec
import time
import os
import warnings

warnings.filterwarnings("ignore")

gpus = range(0,1)
runs = 2
batchsizes = [9] + range(10,17,2) + range(20,250,20)
learnrates=[0.15]
epochs = 1
#iterations = [2,4,8,16,32,64,128]
tasks = []
logdir = "logs/cifar_VGG_timings_and_profiles/deterministic/"
algo=1
with_profiling = True
logfilebase = "cifar_VGG_determ"
command = "python chainer/examples/cifar/deterministic/train_cifar_determ.py -d cifar100"
if not os.path.exists(logdir):
    os.makedirs(logdir)
print "Logdir",logdir

#for iterate in iterations:
for run in range(runs):
    for batch in batchsizes:
        for lr in learnrates:
            logname = "{}_b{}_l{}_{:01d}".format(logfilebase,batch,lr,run)
            logfile = os.path.join(logdir,"{}.log".format(logname))
            command_pars = command+" -e {} -b {} -l {} ".format(epochs,batch,lr)
            if os.path.isfile(logfile):
                print "file",logfile,"exists."
            else:
                task = {"comm":command_pars,"logfile":logfile,"batch":batch,"nvsmi":True}
                tasks.append(task)

            if with_profiling:
                logfile = os.path.join(logdir,"{}.nvprof".format(logname))
                if os.path.isfile(logfile):
                    print "file",logfile,"exists."
                else:
                    profcommand = "nvprof  -u s --profile-api-trace none --unified-memory-profiling off --csv --log-file {} {}".format(logfile,command_pars)
                    task = {"comm":profcommand,"logfile":logfile,"batch":batch,"nvsmi":False}
                    tasks.append(task)

print "Have",len(tasks),"tasks"
gpu = -1
for i in range(0,len(tasks)):
    gpu = multigpuexec.getNextFreeGPU(gpus, start=gpu+1,c=4,d=1,nvsmi=tasks[i]["nvsmi"],mode="dmon")
    gpu_info = multigpuexec.getGPUinfo(gpu)
    f = open(tasks[i]["logfile"],"w+")
    f.write("command:{}\n".format(tasks[i]["comm"]))
    f.write("b{}\n".format(tasks[i]["batch"]))
    f.write("GPU: {}\n".format(gpu_info))
    f.close()
    multigpuexec.runTask(tasks[i],gpu,nvsmi=tasks[i]["nvsmi"],delay=2)
    print "{}/{} tasks".format(i+1,len(tasks))
    time.sleep(1)


