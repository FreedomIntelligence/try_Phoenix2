from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

# lightning deepspeed has saved a directory instead of a file
save_path = "lightning_logs/version_0/checkpoints/epoch=0-step=0.ckpt/"
output_path = "lightning_model.pt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)

# 注意这个合并不包含lr和scheduler https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html#collating-single-file-checkpoint-for-deepspeed-zero-stage-3