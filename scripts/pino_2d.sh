python main.py \
    --model_name=pino_2d \
    --loss_cfg=/workdir/PISegFull/cfgs/nopde.yaml \
    --metric_cfg=/workdir/PISegFull/cfgs/nopde-metrics.yaml \
    --logdir=/workdir/PISegFull/runs/ \
    --num_cat=2 \
    --optim_lr=1e-3 \
    --lrschedule=warmup_cosine \
    --data_dir=/dataset/CXR/leftlungSZPreprocess/ \
    --deterministic=1 \
    --max_epochs=2000 \
    --seed=0 \
    --batch_size=16 \
    --val_every=10 \
    --warmup_epochs=10 \
    --outsdf=1 \
    --workers=8 

