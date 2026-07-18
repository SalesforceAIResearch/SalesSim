export VLLM_LOGGING_LEVEL=DEBUG
export LOG_VLLM_500_PAYLOAD=1
export LOG_VLLM_500_SUMMARY=1
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

# Note: Update the lora-modules path to point to your local LoRA checkpoint
# and allowed-local-media-path to your SalesSim directory
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-3-4b-it \
      --host 0.0.0.0 --port 8040 --tensor-parallel-size 1 --gpu-memory-utilization 0.9 \
      --max-model-len 32192 --served-model-name gemma_with_reasoning \
      --allowed-local-media-path $(pwd) --tool-call-parser hermes --enable-auto-tool-choice
