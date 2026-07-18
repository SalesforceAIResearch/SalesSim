export VLLM_LOGGING_LEVEL=DEBUG
export LOG_VLLM_500_PAYLOAD=1
# export ENABLE_IMAGE_URL_CACHE=1
# export IMAGE_URL_MAX_DIM=384
export LOG_VLLM_500_SUMMARY=1

CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-VL-8B-Instruct  --host 0.0.0.0 --port 8040 --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --max-model-len  48000 --served-model-name qwen_baseline_with_reasoning --tool-call-parser hermes --enable-auto-tool-choice 
