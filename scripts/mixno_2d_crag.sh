python main.py \
    --model_name=mixno_2d \
    --loss_cfg=/workdir/PISegFull/cfgs/mixno.yaml \
    --metric_cfg=/workdir/PISegFull/cfgs/seg-metrics.yaml \
    --logdir=/workdir/PISegFull/runs/crag/256-800 \
    --num_cat=2 \
    --num_train=1494 \
    --optim_lr=1e-3 \
    --lrschedule=warmup_cosine \
    --traindata_dir=/dataset/CRAG/CRAGPreprocess_256x256/ \
    --valdata_dir=/dataset/CRAG/CRAGPreprocess_800x800/ \
    --deterministic=1 \
    --max_epochs=1000 \
    --seed=2 \
    --low_res=256\
    --batch_size=16 \
    --val_every=10 \
    --warmup_epochs=10 \
    --outsdf=1