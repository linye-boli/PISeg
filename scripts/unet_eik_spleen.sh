python main.py \
    --model_name=unetfa \
    --freeze_backbone=0 \
    --coords=1 \
    --eik=1 \
    --outsdf=1 \
    --tanh=0 \
    --val_only=0\
    --loss_cfg=/workdir/PISeg/cfgs/eik.yaml \
    --logdir=/workdir/PISeg/runs/spleen/unet-eik \
    --batch_size=4 \
    --num_cat=2 \
    --optim_lr=1e-3 \
    --lrschedule=warmup_cosine \
    --rebalance=1 \
    --infer_overlap=0.5 \
    --data_dir=/dataset/MSD/SpleenPreprocess/ \
    --deterministic=1 \
    --max_epochs=5000 \
    --warmup_epochs=0 \
    --seed=0 \
    --roi_x=128 \
    --roi_y=128 \
    --roi_z=48 \
    --device=cuda:0 \
    --val_every=10