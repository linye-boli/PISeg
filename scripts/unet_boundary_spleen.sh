python main.py \
    --model_name=unet \
    --coords=0 \
    --eik=0 \
    --outsdf=0 \
    --tanh=0 \
    --loss_cfg=/workdir/segoperator/cfgs/boundary.yaml \
    --logdir=/workdir/segoperator/runs/spleen/unet-boundary \
    --batch_size=4 \
    --num_cat=2 \
    --optim_lr=1e-3 \
    --lrschedule=warmup_cosine \
    --rebalance=1 \
    --infer_overlap=0.5 \
    --data_dir=/dataset/MSD/SpleenPreprocess/ \
    --deterministic=1 \
    --max_epochs=2000 \
    --seed=0 \
    --roi_x=128 \
    --roi_y=128 \
    --roi_z=48 \
    --device=cuda:0 \