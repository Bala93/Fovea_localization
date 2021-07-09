MODEL='resnet34'

CHECKPOINT='/media/htic/NewVolume3/Balamurali/baby_project/fovea-ets/experiments/'${MODEL}'/best_model.pt'
OUT_DIR='/media/htic/NewVolume3/Balamurali/baby_project/fovea-ets/experiments/'${MODEL}

BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH='/media/htic/NewVolume3/Balamurali/baby_project/fovea-ets/multi-modality_images'
EVAL_CSV_PATH='/media/htic/NewVolume3/Balamurali/baby_project/fovea-ets/test.csv'
POS_PATH='/media/htic/NewVolume3/Balamurali/baby_project/fovea-ets/fovea_localization_training_GT.csv'

echo python test.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --eval-csv-path ${EVAL_CSV_PATH} --pos-csv-path ${POS_PATH} 
python test.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --eval-csv-path ${EVAL_CSV_PATH} --pos-csv-path ${POS_PATH} 
