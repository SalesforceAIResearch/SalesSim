#!/usr/bin/env python3

import argparse
import asyncio
import yaml
import logging
import os
import json
from simulation_utils import (
    enrich_results_with_ideal_recommendations,
    load_scenarios_from_yaml,
    save_results,
    run_batch_simulations
)
from services.service_manager import ServiceManager

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
    if not config.get('scenarios_path'):
        required_fields.append('scenarios_path')

    if required_fields:
        error_msg = f"Missing required configuration fields: {', '.join(required_fields)}"
        logging.error(error_msg)
        raise ValueError(error_msg)

def extract_client_config(model_config):
    """Extract client configuration from model configuration."""
    client_config = {}
    model_params = {}

    for key, value in model_config.items():
        if key in ['api_key', 'organization', 'base_url', 'custom_api_key', 'custom_api_key_env', 'extra_headers']:
            if value is not None:
                client_config[key] = value
        elif key in ['model_name', 'temperature', 'max_tokens']:
            model_params[key] = value

    return client_config, model_params


async def main():
    service_manager = ServiceManager()
    parser = argparse.ArgumentParser(description='Run salesbot-shopperbot simulations')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--save', type=str, required=True,
                       help='Save results to specified JSON file')

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
    validate_config(config)
    ai_customer_model_config = config.get('ai_customer_model', {})
    sales_agent_model_config = config.get('sales_agent_model', {})

    customer_client_config, customer_model_params = extract_client_config(ai_customer_model_config)
    salesbot_client_config, salesbot_model_config = extract_client_config(sales_agent_model_config)

    max_turns = config.get('max_turns', 9)
    scenarios_path = config.get('scenarios_path')

    scenarios_config = None
    if scenarios_path:
        scenarios_config = load_scenarios_from_yaml(scenarios_path)

    # Start services before running simulations
    print("Starting services...")
    if not await service_manager.start_all_services():
        print("Failed to start all services. Exiting.")
        return
    print("All services started successfully.")

    try:
        results = await run_batch_simulations(
            max_turns=max_turns,
            scenarios_config=scenarios_config,
            customer_model_config=customer_model_params,
            salesbot_model_config=salesbot_model_config,
            customer_client_config=customer_client_config,
            salesbot_client_config=salesbot_client_config,
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
            os.makedirs(arguments.save, exist_ok=True)
            results = enrich_results_with_ideal_recommendations(results)
            save_results(results, os.path.join(arguments.save, "results.json"))
            config_save_path = os.path.join(arguments.save, "config.json")
            with open(config_save_path, "w") as f:
                json.dump(config, f, indent=2)

        # Also write simulation results with outcome="error" to file
        error_outcomes = [r for r in results if r.get('outcome') == 'error']
        if error_outcomes:
            from datetime import datetime
            error_results_file = arguments.save.replace('.json', '_error_simulations.log') if arguments.save else 'error_simulations.log'

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
