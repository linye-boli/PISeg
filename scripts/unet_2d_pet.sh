python main.py \
    --model_name=unet_2d \
    --loss_cfg=/workdir/PISegFull/cfgs/mixno.yaml \
    --metric_cfg=/workdir/PISegFull/cfgs/no-metrics.yaml \
    --logdir=/workdir/PISegFull/runs/pet/128-512 \
    --num_cat=2 \
    --num_train=5308 \
    --optim_lr=1e-3 \
    --lrschedule=warmup_cosine \
    --traindata_dir=/dataset/Oxford-III_PET/PETPreprocess_128x128/ \
    --valdata_dir=/dataset/Oxford-III_PET/PETPreprocess_512x512/ \
    --deterministic=1 \
    --max_epochs=500 \
    --seed=2 \
    --batch_size=16 \
    --val_every=10 \
    --warmup_epochs=10 \
    --outsdf=0 \

