python main.py \
    --model_name=unet \
    --tanh=1 \
    --outsdf=1 \
    --loss_cfg=/workdir/segoperator/cfgs/sdm.yaml \
    --logdir=/workdir/segoperator/runs/spleen/unet-sdm \
    --batch_size=8 \
    --num_cat=2 \
    --optim_lr=1e-3 \
    --lrschedule=warmup_cosine \
    --infer_overlap=0.5 \
    --data_dir=/dataset/MSD/SpleenPreprocess/ \
    --deterministic=1 \
    --max_epochs=1000 \
    --seed=0 \
    --roi_x=128 \
    --roi_y=128 \
    --roi_z=48 \