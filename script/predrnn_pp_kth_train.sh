export CUDA_VISIBLE_DEVICES=0
cd ..
python -u train.py \
    --dataset_name action \
    --train_data_paths ../kth_action \
    --valid_data_paths ../kth_action \
    --save_dir checkpoints/kth_predrnn_pp \
    --gen_frm_dir results/kth_predrnn_pp \
    --model_name predrnn_pp \
    --reverse_input 1 \
    --img_width 128 \
    --img_channel 1 \
    --input_length 10 \
    --seq_length 20 \
    --num_hidden 128,64,64,64 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --lr 0.0003 \
    --batch_size 2 \
    --max_iterations 80000 \
    --display_interval 10 \
    --test_interval 2000 \
    --snapshot_interval 5000
#    --pretrained_model ./model.ckpt-15000