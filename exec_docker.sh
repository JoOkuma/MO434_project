docker run -v${PWD}:/tmp/project \
    -v/home/jbragantini/Softwares/TACO:/tmp/TACO \
    -it --rm --ipc=host --gpus=all mo434 \
    bash -c "cd project; python3 train.py"

