BATCH_SIZE=4
NUM_EPOCHS=50
DEVICE='cuda:0'
MODEL='unet'

EXP_DIR=''${MODEL}

DATA_PATH=''
POS_PATH=''

TRAIN_CSV_PATH=''
VALID_CSV_PATH=''

python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --data-path ${DATA_PATH} --pos-csv-path ${POS_PATH --train-csv-path ${TRAIN_CSV_PATH} --valid-csv-path ${VALID_CSV_PATH}