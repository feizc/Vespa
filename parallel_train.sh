torchrun --nnodes=2 --nproc_per_node=8 --node_rank $RANK --rdzv-id=vespa \
--rdzv-endpoint=10.0.13.24:12345 \
accelerate_train.py \
--model VeSpa-H/2 \
--dataset-type wds \
--image-only True \
--data-path /maindata/data/shared/multimodal/public/dataset_gen/mj580w_wds2 \
--anna-path /maindata/data/shared/multimodal/public/dataset_gen/mj580w_wds2 \
--image-size 512 \
--text_encoder_type t5 \
--global-batch-size 16 \
--epochs 20 \
--warmup_epochs 0 \
--accum_iter 8 \
--eval_steps 10 \
--lr 1e-4 \
--latent_space True \
--global-seed 42
