python main.py \
    --model_name=unet \
    --loss_cfg=/workdir/PISeg/cfgs/dice.yaml \
    --logdir=/workdir/PISeg/runs/spleen/unet-dice \
    --batch_size=4 \
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
    --warmup_epochs=0 \