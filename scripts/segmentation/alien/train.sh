python3.9 train_seg.py train \
  --dataroot ./datasets/alien_small/ \
	--weight_decay 0.05 --optim adamw \
	--lr 1e-4 --n_epoch 100 --gamma 0.1 \
	--batch_size 2 --heads 6 --patch_size 64 \
	--dim 768 --encoder_depth 12 \
	--decoder_depth 6 --decoder_dim 512 --decoder_num_heads 16 \
	--channel 10 --augment_scale \
	--name "alien" --face_pos --lw1 1 --lw2 1 \
	--dataset_name alien --seg_parts 4
