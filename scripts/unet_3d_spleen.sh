python main.py \
    --model_name=unet_3d \
    --loss_cfg=/workdir/PISeg/PISeg/cfgs/no.yaml \
    --metric_cfg=/workdir/PISeg/PISeg/cfgs/no-metrics.yaml \
    --logdir=/workdir/PISeg/PISeg/runs/ \
    --num_cat=2 \
    --optim_lr=1e-3 \
    --lrschedule=warmup_cosine \
    --traindata_dir=/dataset/MSD/SpleenPreprocess_64x64x64/ \
    --valdata_dir=/dataset/MSD/SpleenPreprocess_128x128x128/ \
    --deterministic=1 \
    --max_epochs=2000 \
    --seed=0 \
    --batch_size=4 \
    --val_every=10 \
    --warmup_epochs=10 \
    --outsdf=1
