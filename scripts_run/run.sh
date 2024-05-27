CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_lora_sft_v2.yaml

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/opt-125m_lora_sft.yaml

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/opt-125m_lora_pretrain.yaml


CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/full_multi_gpu/opt-125m_full_sft.yaml

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/opt-125m_full_sft.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/opt-1.3b_full_sft.yaml | tee log/opt-1.3b_full_sft.log 2>&1

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_single_gpu/llama3_full_sft.yaml | tee log/llama3_full_sft.log 2>&1




