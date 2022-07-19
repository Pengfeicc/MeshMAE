python3.9 train_cls.py test \
	--dataroot ./datasets/Manifold40-MAPS-96-3/ \
	--batch_size 32 --augment_scale --n_classes 40 \
	--channels 10 --patch_size 64 \
  --n_epoch 1 --name "manifoldBase" \
	--lr_milestones "30 60"  --optim "adamw" \
	--weight_decay 0.1 \
	--lr 1e-4 \
	--depth 12 \
	--heads 12 \
	--encoder_depth 12 \
	--decoder_depth 6 \
	--decoder_dim 512 \
	--decoder_num_heads 16 \
  --checkpoint "./checkpoints/ModelNet40.pkl" \
	--num_warmup_steps "2" \
	--dim 768
