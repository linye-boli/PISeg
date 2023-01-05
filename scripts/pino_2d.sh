python main.py \
    --model_name=pino_2d \
    --loss_cfg=/workdir/PISeg/PISeg/cfgs/nopde.yaml \
    --metric_cfg=/workdir/PISeg/PISeg/cfgs/no-metrics.yaml \
    --logdir=/workdir/PISeg/PISeg/runs/ \
    --num_cat=2 \
    --optim_lr=1e-3 \
    --lrschedule=warmup_cosine \
    --traindata_dir=/dataset/CXR/leftlungSZPreprocess_64x64/ \
    --valdata_dir=/dataset/CXR/leftlungSZPreprocess_128x128/ \
    --deterministic=1 \
    --max_epochs=2000 \
    --seed=0 \
    --batch_size=16 \
    --val_every=10 \
    --warmup_epochs=10 \
    --outsdf=1 \
    --workers=8 \
    --device='cuda:1'

