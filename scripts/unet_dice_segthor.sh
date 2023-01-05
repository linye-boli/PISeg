python main.py \
    --model_name=unet \
    --batch_size=8 \
    --logdir=/workdir/segoperator/runs/segthor/ \
    --num_cat=5 \
    --optim_lr=1e-3 \
    --lrschedule=warmup_cosine \
    --infer_overlap=0.5 \
    --data_dir=/dataset/segthor/segthorPreprocess/ \
    --deterministic=1 \
    --max_epochs=2000 \
    --seed=0 \
    --roi_x=160 \
    --roi_y=160 \
    --roi_z=128