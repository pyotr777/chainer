#!/usr/bin/env python

# Prepares and runs multiple tasks on multiple GPUs: one task per GPU.
# Waits if no GPUs available. For GPU availability check uses "nvidia-smi dmon" command.

# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

import multigpuexec
import time
import os

gpus = range(0,1)
runs = 1
batchsizes = range(12,1225,2)
learnrates=[0.15]
epochs=1
tasks = []
logdir = "logs/microseries/"
# command = "python chainer/examples/cifar/train_cifar_model.py -d cifar100 --model model128"
command = "python chainer/examples/cifar/train_cifar_model.py -d cifar100 --model VGGb --iterations 2"
if not os.path.exists(logdir):
    os.makedirs(logdir)
print "Logdir",logdir

for run in range(runs):
    for batch in batchsizes:
        for lr in learnrates:
            logfile=os.path.join(logdir,"cifar_log_b{}_l{:.3f}_{:04d}.log".format(batch,lr,run))
            if os.path.isfile(logfile):
                print "file",logfile,"exists."
                continue

            task = {"comm":"{} -e {} -b {} -l {} ".format(command,epochs,batch,lr),"logfile":logfile,"batch":batch,"lr":lr}
            tasks.append(task)

print "Have",len(tasks),"tasks"
gpu = -1
for i in range(0,len(tasks)):
    gpu = multigpuexec.getNextFreeGPU(gpus, start=gpu+1,c=4,d=1,nvsmi=True)
    gpu_info = multigpuexec.getGPUinfo(gpu)
    f = open(tasks[i]["logfile"],"w+")
    f.write("b{} l{}\n".format(tasks[i]["batch"],tasks[i]["lr"]))
    f.write("GPU: {}\n".format(gpu_info))
    f.close()
    print "{}/{} tasks".format(i+1,len(tasks))
    multigpuexec.runTask(tasks[i],gpu,nvsmi=True)
    time.sleep(0)


