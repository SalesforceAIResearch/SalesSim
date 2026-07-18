#!/usr/bin/env python3

import argparse
import asyncio
import yaml
import logging
import os
import json
from datetime import datetime
from salessim.simulation_utils import (
    run_batch_simulations
)
from salessim.services.service_manager import ServiceManager


def setup_error_logging(output_dir):
    """Set up file logging for VLLM and sales service errors."""
    if not output_dir:
        return None

    os.makedirs(output_dir, exist_ok=True)
    error_log_path = os.path.join(output_dir, "service_errors.log")

    # Create a file handler for error logging
    file_handler = logging.FileHandler(error_log_path)
    file_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to loggers that handle VLLM and sales service errors
    loggers_to_monitor = [
        'common.ai_client',           # VLLM errors
        'salessim.services.http_clients',  # Sales service HTTP errors
        'salessim.services.sales_service', # Sales service errors
    ]

    for logger_name in loggers_to_monitor:
        logger = logging.getLogger(logger_name)
        logger.addHandler(file_handler)
        logger.setLevel(logging.ERROR)

    # Also capture root logger errors related to these services
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    logging.info(f"Service error logging enabled: {error_log_path}")
    return error_log_path

async def cancel_all_tasks():
    # Get all tasks running in the current event loop
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    if not tasks:
        print("No pending tasks to cancel.")
        return

    print(f"Cancelling {len(tasks)} tasks...")

    # Cancel all the tasks
    for task in tasks:
        task.cancel()

    # Wait for all canceled tasks to finish (handle the CancelledError)
    # The return_when=asyncio.ALL_COMPLETED ensures we wait for all of them.
    await asyncio.gather(*tasks, return_exceptions=True)


def load_config_from_yaml(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        raise

def validate_config(config):
    """Validate that all required configuration fields are present."""
    required_fields = []

    # Check ai_customer_model.model_name
    ai_customer_model = config.get('ai_customer_model')
    if not ai_customer_model:
        required_fields.append('ai_customer_model')
    elif not ai_customer_model.get('model_name'):
        required_fields.append('ai_customer_model.model_name')

    # Check sales_agent_model.model_name
    sales_agent_model = config.get('sales_agent_model')
    if not sales_agent_model:
        required_fields.append('sales_agent_model')
    elif not sales_agent_model.get('model_name'):
        required_fields.append('sales_agent_model.model_name')

    # Check max_turns
    if config.get('max_turns') is None:
        required_fields.append('max_turns')

    # Check scenarios_path
    # if not config.get('scenarios_path'):
    #     required_fields.append('scenarios_path')

    if required_fields:
        error_msg = f"Missing required configuration fields: {', '.join(required_fields)}"
        logging.error(error_msg)
        raise ValueError(error_msg)

def apply_product_category_placeholders(config, product_category):
    """Replace {product_category} placeholders in config values."""
    if not product_category:
        return config

    def replace_value(value):
        if isinstance(value, str):
            return value.replace("{product_category}", product_category)
        if isinstance(value, list):
            return [replace_value(v) for v in value]
        if isinstance(value, dict):
            return {k: replace_value(v) for k, v in value.items()}
        return value

    return replace_value(config)

def apply_runtime_config(config):
    """Apply optional runtime settings from YAML into environment variables."""
    env_map = {
        "products_index_dir": "PRODUCTS_INDEX_DIR",
        "guides_index_dir": "GUIDES_INDEX_DIR",
        "enable_product_images": "ENABLE_PRODUCT_IMAGES",
        "product_image_dir": "PRODUCT_IMAGE_DIR",
        "prune_previous_image_urls": "PRUNE_PREVIOUS_IMAGE_URLS",
        "lookup_cuda_visible_devices": "LOOKUP_CUDA_VISIBLE_DEVICES",
        "lookup_base_url": "LOOKUP_BASE_URL",
        "nltk_data_dir": "NLTK_DATA",
    }
    for key, env_key in env_map.items():
        if key not in config:
            continue
        value = config.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            os.environ[env_key] = "1" if value else "0"
        elif isinstance(value, list):
            os.environ[env_key] = ",".join(str(v) for v in value)
        else:
            os.environ[env_key] = str(value)

def validate_lookup_assets(config):
    products_index_dir = config.get("products_index_dir")
    guides_index_dir = config.get("guides_index_dir")
    products_data_dir = "data/products"

    if products_index_dir and not os.path.isdir(products_index_dir):
        logging.warning(
            "products_index_dir not found: %s. Lookup service will try to build a new index.",
            products_index_dir,
        )
        if not os.path.isdir(products_data_dir):
            logging.warning(
                "%s not found; lookup service may have no data to build an index.",
                products_data_dir,
            )
        else:
            json_files = [
                f for f in os.listdir(products_data_dir) if f.endswith(".json")
            ]
            if not json_files:
                logging.warning(
                    "No .json files in %s; lookup service may have no data to build an index.",
                    products_data_dir,
                )

    if guides_index_dir and not os.path.isdir(guides_index_dir):
        logging.warning(
            "guides_index_dir not found: %s. Lookup service will try to build a new index.",
            guides_index_dir,
        )
        if not os.path.exists("data/guides.json"):
            logging.warning("data/guides.json not found; buying guide lookup may fail.")

def extract_client_config(model_config):
    """Extract client configuration from model configuration."""
    client_config = {}
    model_params = {}

    for key, value in model_config.items():
        if key in ['api_key', 'organization', 'base_url', 'custom_api_key', 'custom_api_key_env', 'extra_headers']:
            if value is not None:
                client_config[key] = value
        elif key in [
            'model_name',
            'temperature',
            'max_tokens',
            'top_p',
            'top_k',
            'thinking_budget',
            'with_thinking',
            'repetition_penalty',
            'frequency_penalty',
            'presence_penalty',
            'stop',
            'reasoning_effort',
        ]:
            model_params[key] = value

    return client_config, model_params

def convert_jsonl_to_json(output_dir):
    """Convert JSONL file to final JSON file."""
    jsonl_file = os.path.join(output_dir, "results.jsonl")
    json_file = os.path.join(output_dir, "results.json")

    if os.path.exists(jsonl_file):
        logging.info(f"Converting {jsonl_file} to {json_file}")
        all_results = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line.strip()))

        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"Converted JSONL to JSON: {json_file}")
    else:
        logging.warning(f"JSONL file not found: {jsonl_file}")


async def main():
    service_manager = ServiceManager()
    parser = argparse.ArgumentParser(description='Run salesbot-shopperbot simulations')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--sales-agent-config', type=str,
                       default='eval_envs/sales_agent_config.yaml',
                       help='Path to separate sales agent YAML configuration file')
    parser.add_argument('--products-config', type=str,
                       default='eval_envs/products_env.yaml',
                       help='Path to shared products environment YAML configuration file')
    parser.add_argument('--save', type=str, required=True,
                       help='Save results to specified JSON file')
    parser.add_argument('--product-category', type=str, default=None,
                       help='Product category (e.g. laptops, female_clothing)')

    parser.add_argument('--list-model-examples', action='store_true',
                       help='Show examples of supported model formats and exit')

    arguments = parser.parse_args()
    if arguments.list_model_examples:
        print("Supported model formats:")
        print("  OpenAI: gpt-4-turbo, gpt-4, gpt-3.5-turbo")
        print("  Anthropic: anthropic/claude-3-sonnet, anthropic/claude-3-haiku")
        print("  GCP Vertex AI: vertex_ai/claude-3-sonnet@20240229")
        print("  Azure OpenAI: azure/gpt-4")
        print("  Ollama: ollama/llama2, ollama/mistral")
        print("  Hugging Face: huggingface/microsoft/DialoGPT-medium")
        print("\nFor more providers, see: https://docs.litellm.ai/docs/providers")
        return

    config = load_config_from_yaml(arguments.config)
    sales_agent_config = load_config_from_yaml(arguments.sales_agent_config)
    products_config = load_config_from_yaml(arguments.products_config)
    config['sales_agent_model'] = sales_agent_config.get('sales_agent_model', {})

    # Merge products config into main config (main config can override)
    for key in ['guides_index_dir', 'products_index_dir', 'enable_product_images',
                'prune_previous_image_urls', 'lookup_cuda_visible_devices',
                'lookup_base_url', 'nltk_data_dir']:
        if key not in config or config.get(key) is None:
            config[key] = products_config.get(key)

    # Store product_categories mapping for later use
    product_categories_config = products_config.get('product_categories', {})
    product_category = arguments.product_category or config.get("product_category")
    if not product_category:
        products = config.get("products")
        if isinstance(products, list) and len(products) == 1:
            product_category = products[0]
    if product_category:
        assert product_category in ["laptops", "female_clothing", "male_clothing", "smart_watch", "game_gadgets"], (
            "product_category must be one of ['laptops', 'female_clothing', 'male_clothing', 'smart_watch', 'game_gadgets']"
        )
    config = apply_product_category_placeholders(config, product_category)

    # Build persona_files from product_categories_config
    products_list = config.get('products', [])
    if products_list and product_categories_config:
        # Build persona_files mapping from per-category config if not already set
        if not config.get('persona_files'):
            persona_mapping = {}
            for prod in products_list:
                cat_config = product_categories_config.get(prod, {})
                if cat_config.get('persona_file'):
                    persona_mapping[prod] = cat_config['persona_file']
            if persona_mapping:
                config['persona_files'] = persona_mapping

        # Build product_image_dirs mapping from per-category config
        if not config.get('product_image_dirs'):
            image_dirs = {}
            for prod in products_list:
                cat_config = product_categories_config.get(prod, {})
                if cat_config.get('image_dir'):
                    image_dirs[prod] = cat_config['image_dir']
            if image_dirs:
                config['product_image_dirs'] = image_dirs

    validate_config(config)
    apply_runtime_config(config)
    validate_lookup_assets(config)
    ai_customer_model_config = config.get('ai_customer_model', {})
    sales_agent_model_config = config.get('sales_agent_model', {})

    customer_client_config, customer_model_params = extract_client_config(ai_customer_model_config)
    salesbot_client_config, salesbot_model_config = extract_client_config(sales_agent_model_config)

    max_turns = config.get('max_turns', 9)
    num_runs = config.get('num_runs', 1)
    products = config.get('products')
    persona_files = config.get('persona_files', {})
    product_image_dir = config.get('product_image_dir')
    product_image_dirs = config.get('product_image_dirs', {})

    scenarios_config = None
    if products:
        print("loading from products")
        scenarios_config = products
    # Start services before running simulations
    print("Starting services...")
    if not await service_manager.start_all_services():
        print("Failed to start all services. Exiting.")
        return
    print("All services started successfully.")

    results = None
    error_log_path = None
    try:
        # Prepare streaming output file path
        if arguments.save:
            if os.path.exists(arguments.save) and len(os.listdir(arguments.save)) == 0:
                os.rmdir(arguments.save)
            assert not os.path.exists(arguments.save), "Save directory already exists, please remove otherwise this will append to the existing file."
            os.makedirs(arguments.save)
            # Set up error logging for VLLM and sales service errors
            error_log_path = setup_error_logging(arguments.save)
            if error_log_path:
                print(f"Service error logging enabled: {error_log_path}")

        results = await run_batch_simulations(
            max_turns=max_turns,
            scenarios_config=scenarios_config,
            customer_model_config=customer_model_params,
            salesbot_model_config=salesbot_model_config,
            customer_client_config=customer_client_config,
            salesbot_client_config=salesbot_client_config,
            output_dir=arguments.save,
            config_path=arguments.config,
            persona_files=persona_files,
            product_image_dir=product_image_dir,
            product_image_dirs=product_image_dirs,
            num_runs=num_runs,
        )
    except Exception as e:
        print(f"Error during simulation: {e}")
        raise e
    finally:
        # Always stop services, even if simulation fails
        print("Stopping services...")
        await service_manager.stop_all_services()
        print("Services stopped.")
        if arguments.save and results:
            config_save_path = os.path.join(arguments.save, "config.json")
            with open(config_save_path, "w") as f:
                json.dump(config, f, indent=2)

            # Convert JSONL to final JSON
            convert_jsonl_to_json(arguments.save)

        # Also write simulation results with outcome="error" to file
        if results:
            error_outcomes = [r for r in results if r.get('outcome') == 'error']
        else:
            error_outcomes = []
        if error_outcomes:
            error_results_file = os.path.join(arguments.save, "error.json")

            with open(error_results_file, 'w') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - Total simulations with error outcome: {len(error_outcomes)}\n")
                for i, result in enumerate(error_outcomes, 1):
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - Error simulation {i}:\n")
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR -   Conversation ID: {result.get('conversation_id', 'N/A')}\n")
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR -   Total turns: {result.get('total_turns', 'N/A')}\n")
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR -   Timestamp: {result.get('timestamp', 'N/A')}\n")
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR -   Shopper persona: {result.get('shopper_persona', 'N/A')}\n")
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR -   Error message: {result.get('error_message', 'N/A')}\n")
                    # Write last few conversation entries if available
                    conversation = result.get('conversation', [])
                    if conversation:
                        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR -   Last conversation entries:\n")
                        for entry in conversation[-3:]:  # Last 3 entries
                            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR -     Turn {entry.get('turn', 'N/A')}: {entry.get('speaker', 'N/A')}: {entry.get('text', 'N/A')[:100]}...\n")

            print(f"Error simulations written to: {error_results_file}")

        await cancel_all_tasks() # LiteLLM has issues with closing async loggerworker. 

if __name__ == "__main__":
    asyncio.run(main())
