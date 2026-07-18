source /opt/conda/etc/profile.d/conda.sh
conda activate glm_env # need transforme>=5.0.0rc0 and vllm>=0.12.0

vllm serve zai-org/GLM-4.6V-Flash \
      --host 0.0.0.0 --port 8040 \
      --tensor-parallel-size 4 \
      --tool-call-parser glm45 \
      --enable-auto-tool-choice \
      --allowed-local-media-path / \
      --mm-encoder-tp-mode data \
      --mm-processor-cache-type shm \
      --max-model-len 131072 \
      --max-num-batched-tokens 32768 \
      --gpu-memory-utilization 0.95 \
      --served-model-name glm46_with_reasoning

