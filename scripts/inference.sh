export PYTHONPATH=.:$PYTHONPATH
ACC_CONFIG_FILE="configs/acc_configs/multi_default.yaml"
export CUDA_VISIBLE_DEVICES="0"
NUM_PROCESSES=2
MASTER_PORT=29500

OUTPUT_DIR="out"
mkdir -p $OUTPUT_DIR
cat "$0" >> $OUTPUT_DIR/run_script.sh

DATA_DIR=data/iHarmony4
TEST_FILE=test.jsonl

accelerate launch --config_file $ACC_CONFIG_FILE --num_processes $NUM_PROCESSES --main_process_port $MASTER_PORT \
scripts/inference/main.py \
    --pretrained_model_name_or_path "/kaggle/working/DiffHarmony/checkpoints/diffharmonry-checkpoint/base" \
    --pretrained_vae_model_name_or_path "/kaggle/working/DiffHarmony/checkpoints/diffharmonry-checkpoint/condition_vae" \
    --pretrained_unet_model_name_or_path "/kaggle/working/DiffHarmony/checkpoints/diffharmonry-checkpoint/base/unet" \
    --dataset_root $DATA_DIR \
	--test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
	--seed=0 \
	--resolution=512 \
	--output_resolution=256 \
	--eval_batch_size=1 \
	--dataloader_num_workers=4 \
	--mixed_precision="fp16"

	# --stage2_model_name_or_path ""