docker run -v${PWD}:/tmp/project \
    -v/data/home/jordao/MO434/TACO:/tmp/TACO \
    --name mo434 \
    -it --rm --ipc=host --gpus=all mo434 \
    bash -c "cd project; python3 train.py"

