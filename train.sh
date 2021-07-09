BATCH_SIZE=4
NUM_EPOCHS=50
DEVICE='cuda:0'
MODEL='unet'

EXP_DIR='/media/htic/NewVolume3/Balamurali/baby_project/fovea-ets/experiments/'${MODEL}

DATA_PATH='/media/htic/NewVolume3/Balamurali/baby_project/fovea-ets/multi-modality_images'
POS_PATH='/media/htic/NewVolume3/Balamurali/baby_project/fovea-ets/fovea_localization_training_GT.csv'

TRAIN_CSV_PATH='/media/htic/NewVolume3/Balamurali/baby_project/fovea-ets/train.csv'
VALID_CSV_PATH='/media/htic/NewVolume3/Balamurali/baby_project/fovea-ets/val.csv'



echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --data-path ${DATA_PATH} --pos-csv-path ${POS_PATH} --train-csv-path ${TRAIN_CSV_PATH} --valid-csv-path ${VALID_CSV_PATH}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --data-path ${DATA_PATH} --pos-csv-path ${POS_PATH} --train-csv-path ${TRAIN_CSV_PATH} --valid-csv-path ${VALID_CSV_PATH}
