#!/bin/bash
# Start docker container
IMAGE="nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04"
comm="nvidia-docker run -t --rm -v $(pwd):/root/ -w /root $IMAGE /root/init_container.sh $@"
echo "NV_GPU=$NV_GPU"
echo "Runnning command $comm"
$comm 2>&1
echo "Container stopped."
#docker logs $cont