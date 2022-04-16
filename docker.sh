!/bin/bash

CONTAINER_NAME=dsfd
IMAGES=yuta0514/dsfd
TAGS=1.8
PORT=8888

docker run --rm -it --gpus all --ipc host -v ~/dataset:/mnt -v $PWD:$PWD -p ${PORT}:${PORT} --name ${CONTAINER_NAME} ${IMAGES}:${TAGS}
