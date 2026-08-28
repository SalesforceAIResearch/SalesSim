# CustomerSim: Multimodal LLMs as Retail User Simulators

CustomerSim is an environment for retail simulations ([paper](https://huggingface.co/Salesforce/RetailSim-GRPO-4B)) to support benchmarking and development of customer simulators. 

![](visual_customersim.png)

## Benchmark Numbers

| Method | Female | Male | Smart Watch | Laptops | Games | Overall |
|--------|--------|------|-------------|---------|-------|---------|
| Gemma3 (4B) | 0.425 | 0.342 | 0.386 | 0.378 | 0.230 | 0.352 |
| w/ HS | 0.356 | 0.195 | 0.265 | 0.333 | 0.5 | 0.330 |
| + SFT | — | 0.532 | 0.578 | 0.593 | 0.790 | 0.623 |
| + (SFT + UserGRPO) | — | 0.571 | 0.6 | 0.741 | 0.8 | 0.678 |
| GPT-5.5 | 0.692 | 0.556 | 0.831 | 0.571 | 0.790 | 0.723 |
| GLM-4.6V-Flash (9B) | 0.48 | 0.416 | 0.483 | 0.556 | 0.65 | 0.517 |
| w/ HS | 0.384 | 0.481 | 0.169 | 0.296 | 0.61 | 0.388 |
| Qwen3-VL-8B-Instruct | 0.507 | 0.506 | 0.558 |  0.37 | 0.460 | 0.480 |
| w/ HS | 0.514 | 0.494 | 0.537 | 0.33 | 0.550 | 0.485 |

We also introduce a RL recipe for improving fidelity named UserGRPO, with model weights [here](https://huggingface.co/Salesforce/RetailSim-GRPO-4B). On Gemma-3-4B, combining SFT with UserGRPO significantly improves performance.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/customersim.git
cd customersim

# Install dependencies
pip install -e .
```

## Quick Start

1. Start the vLLM server using one of the provided scripts:

```bash
bash run_configs/gemma_with_reasoning/run_vllm_script.sh
```

This will start a vLLM server on `http://127.0.0.1:8040` with the Gemma-3-4B-IT model.

2. Start the sales agent service:

```bash
python -m customersim.agents.service --config run_configs/sales_agent_config.yaml
```

3. Run the simulation:

```bash
python -m customersim.simulate \
  --config run_configs/gemma_with_reasoning/experiment.yaml \
  --save output_dir/
```

The experiment configuration will use the vLLM endpoint specified in the config file. You can modify the `base_url` in the experiment.yaml to point to your vLLM server.

See `run_configs/` for more examples including Qwen and GLM configurations.


