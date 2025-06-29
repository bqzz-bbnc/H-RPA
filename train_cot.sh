# 下面CUDA_VISIBLE_DEVICES= 的卡数需要和yaml文件中的ray_num_workers一致
CUDA_VISIBLE_DEVICES=0,1,2,3 USE_RAY=1 llamafactory-cli train configs/train_lora/glm4_lora_sft_ray_cot.yaml

# llamafactory-cli export configs/merge_lora/glm4_lora_sft_ray_cot.yaml