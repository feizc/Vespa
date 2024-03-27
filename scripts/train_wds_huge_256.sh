torchrun --nnodes=1 --nproc_per_node=8 train.py \
--model VeSpa-H/2 \
--dataset-type wds \
--image-only True \
--data-path /maindata/data/shared/multimodal/public/dataset_gen/mj580w_wds2 \
--anna-path /maindata/data/shared/multimodal/public/dataset_gen/mj580w_wds2 \
--image-size 512 \
--text_encoder_type t5 \
--global-batch-size 32 \
--epochs 20 \
--warmup_epochs 0 \
--accum_iter 8 \
--eval_steps 10000000 \
--lr 1e-4 \
--latent_space True \
--global-seed 45 \
--resume /maindata/data/shared/multimodal/zhengcong.fei/code/vespa/results/VeSpa-H-2-wds-image-True/checkpoints/ckpt2.pt 