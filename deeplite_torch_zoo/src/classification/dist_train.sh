GPUS=$1
ARCH=$2
DATASET=$3
DATAROOT=$4
PORT=${PORT:-29507}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_dp.py --world_size $GPUS --arch $ARCH --dataset-name $DATASET --data-root $DATAROOT --launcher pytorch ${@:6}
