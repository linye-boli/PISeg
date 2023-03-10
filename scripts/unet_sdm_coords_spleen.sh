python main.py \
    --model_name=unet_header \
    --outsdf=1 \
    --coords=1 \
    --eik=0 \
    --tanh=1 \
    --loss_cfg=/workdir/segoperator/cfgs/sdm.yaml \
    --logdir=/workdir/segoperator/runs/spleen/unet-sdm-coords \
    --batch_size=4 \
    --num_cat=2 \
    --optim_lr=1e-3 \
    --lrschedule=warmup_cosine \
    --infer_overlap=0.5 \
    --data_dir=/dataset/MSD/SpleenPreprocess/ \
    --deterministic=1 \
    --max_epochs=2000 \
    --seed=4 \
    --roi_x=128 \
    --roi_y=128 \
    --roi_z=48 \
    --device=cuda:3 \