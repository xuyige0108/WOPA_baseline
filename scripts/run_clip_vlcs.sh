#!/bin/bash
# pip install yacs
# pip install gdown

cd ..


DATASET_ROOT=/data2/tyl/datasets
DATASET=vlcs
NET=vitb16_clip
DATASET_YAML=vlcs_source_free
#NET_INIT=pretrain/resnet50.pth

ND=0
BATCH=128
I2_EPOCHS=50


if [ ${DATASET} == 'pacs' ]; then
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch

    DATASET_NAME='PACS_SF'
    
elif [ ${DATASET} == 'office_home_dg' ]; then
    D1=art
    D2=clipart
    D3=product
    D4=real_world
    
    DATASET_NAME='OfficeHomeDG_SF'
    
elif [ ${DATASET} == 'vlcs' ]; then
    D1=CALTECH
    D2=LABELME
    D3=PASCAL
    D4=SUN
    
    DATASET_NAME='VLCS_SF'

fi

for SEED in $(seq 1 3)
do
    CUDA_VISIBLE_DEVICES=6 \
    python train.py \
    --root ${DATASET_ROOT} \
    --seed ${SEED} \
    --use_cuda True \
    --trainer WOPA_clip \
    --source-domains none\
    --target-domains ${D1} ${D2} ${D3} ${D4} \
    --dataset-config-file configs/datasets/dg/${DATASET_YAML}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir output/${DATASET}/NS80_lr5_arcface_E${I2_EPOCHS}_5_0.5_${NET}/seed${SEED} \
    --arcface_s 5\
    --arcface_m 0.5\
    --num_styles 80\
    --txts_path /data2/tyl/WOPA_v3/dassl/txts\
    MODEL.BACKBONE.NAME ${NET} \
    DATALOADER.TRAIN_X.SAMPLER RandomSampler \
    DATALOADER.TRAIN_X.N_DOMAIN ${ND} \
    DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH} \
    OPTIM.MAX_EPOCH ${I2_EPOCHS} \
    DATASET.NAME ${DATASET_NAME} \
    OPTIM.LR 0.005 \
    DATASET.ROOT ${DATASET_ROOT}
    
done
