#!/usr/bin/env python3
from ast import List
import asyncio
import os
import json
import asyncio
import traceback
import copy
import uuid
import hashlib
from datetime import datetime
import shutil
import tqdm
from typing import Union
from salessim.agents.sales_agent.sales_agent import SalesAgent
from salessim.agents.ai_customer.ai_customer import load_personas, CustomerSimulator
from common.ai_client import create_client_from_model_name
from common.bcolors import bcolors


def create_persona_hash(shopper_persona, shopper_preferences):
    """Create a hash from shopper persona and preferences."""
    # Convert all components to strings for consistent hashing
    persona_str = json.dumps(shopper_persona, sort_keys=True) if shopper_persona else ""
    preferences_str = json.dumps(shopper_preferences, sort_keys=True) if shopper_preferences else ""

    # Combine all components
    combined_str = f"{persona_str}|{preferences_str}"

    # Create SHA256 hash
    return hashlib.sha256(combined_str.encode('utf-8')).hexdigest()



def _content_has_image_url(content) -> bool:
    if isinstance(content, list):
        return any(
            isinstance(block, dict) and block.get("type") == "image_url"
            for block in content
        )
    return False


def _strip_image_urls_from_content(content):
    if isinstance(content, list):
        return [
            block
            for block in content
            if not (isinstance(block, dict) and block.get("type") == "image_url")
        ]
    return content

def _build_message(role: str, content: Union[str, dict]) -> dict:
    return {"speaker": role, "content": content}

def _replace_image_urls_with_filenames(entry: dict) -> dict:
    content = entry.get("text")
    if not isinstance(content, list):
        return entry
    image_filenames = []
    for item in entry.get("recommended_items", []) or []:
        metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
        image_path = metadata.get("image")
        if image_path:
            image_filenames.append(os.path.basename(image_path))
    filename_iter = iter(image_filenames)
    replaced = False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image_url":
            image_url = block.get("image_url")
            if isinstance(image_url, dict) and "url" in image_url:
                filename = next(filename_iter, "embedded_image")
                image_url["url"] = filename
                replaced = True
    if replaced:
        entry["content"] = content
    return entry


async def run_simulation(max_turns, shopperbot, salesbot, verbose=True):
    """
    Run a simulation with optional shared AI client.
    """
    try:
        # Initialize conversation
        chat_history = []
        conversation_log = []
        turn_count = 0
        outcome = "incomplete"
        prune_previous_images = os.environ.get("PRUNE_PREVIOUS_IMAGE_URLS") == "1"

        # Start with salesbot greeting
        salesperson_content =  "Hello! I'm here to help you find the perfect product. What are you looking for today?"

        conversation_log.append({
            "speaker": "Salesperson",
            "text": salesperson_content,
            "turn": turn_count
        })

        # Run conversation simulation
        while turn_count < max_turns:
            try:
                turn_count += 1

                # Shopperbot responds
                shopper_response = await shopperbot.async_generate(
                    curr_content=salesperson_content,
                    chat_history=chat_history,
                )

                shopper_content = shopper_response["content"]
                if prune_previous_images and _content_has_image_url(salesperson_content):
                    pruned_history = []
                    for msg in chat_history:
                        pruned_content = _strip_image_urls_from_content(msg["content"])
                        if isinstance(pruned_content, list) and not pruned_content:
                            continue
                        pruned_history.append(
                            {"speaker": msg["speaker"], "content": pruned_content}
                        )
                    chat_history = pruned_history
                chat_history.append(_build_message("Salesperson", salesperson_content))

                if verbose:
                    print(f"{bcolors.OKCYAN}Shopper: {shopper_content}{bcolors.ENDC}")
                shopper_log_entry = {
                    "speaker": "Shopper",
                    "text": shopper_content,
                    "shopper_action": json.dumps(shopper_response["shopper_action"]),
                    "reasoning": shopper_response.get("reasoning", ""),
                    "turn": turn_count,
                    "preferences_used": shopper_response.get("preferences", "")
                }
                conversation_log.append(shopper_log_entry)


                # Check if shopper accepted a recommendation
                if "add_to_cart" in str(shopper_response["shopper_action"]):
                    outcome = "accepted"
                    if verbose:
                        print(f"{bcolors.OKGREEN}Recommendation accepted! Simulation complete.{bcolors.ENDC}")
                    break

                if "end_conversation" in str(shopper_response["shopper_action"]):
                    outcome = "ended_by_shopper"
                    if verbose:
                        print(f"{bcolors.WARNING}Shopper ended the conversation.{bcolors.ENDC}")
                    break

                # Salesagent responds
                sales_response = await salesbot.async_generate(shopper_response["content"], chat_history)
                chat_history.append(_build_message("Shopper", shopper_response["content"]))
                salesperson_content = sales_response["content"]

                salesperson_log_entry = {
                    "speaker": "Salesperson",
                    "text": salesperson_content,
                    "turn": turn_count,
                    "reasoning": sales_response.get("reasoning", ""),
                    "knowledge_used": sales_response.get("knowledge", ""),
                    "recommended_items": sales_response.get("recommended_items", []),
                    "recommended_items_count": len(sales_response.get("recommended_items", []))
                }
                conversation_log.append(salesperson_log_entry)



            except Exception as e:
                if verbose:
                    print(f"{bcolors.FAIL}Error during simulation: {e}{bcolors.ENDC}")
                    print(traceback.format_exc())
                outcome = "error"
                error_message = f"Error during simulation: {traceback.format_exc()}"
                break

        if turn_count >= max_turns and outcome == "incomplete":
            outcome = "max_turns_reached"
            if verbose:
                print(f"{bcolors.WARNING}Simulation ended after {max_turns} turns{bcolors.ENDC}")


        if verbose:
            print(f"\n{bcolors.OKGREEN}Final outcome: {outcome}{bcolors.ENDC}")
        # Generate a unique conversation ID
        conversation_id = str(uuid.uuid4())

        # Create persona hash
        persona_hash = create_persona_hash(
            shopperbot.current_persona,
            shopperbot.all_preferences
        )

        return {
            "conversation_id": conversation_id,
            "shopper_preferences": shopperbot.all_preferences,
            "shopper_persona": shopperbot.current_persona,
            "persona_hash": persona_hash,
            "conversation": conversation_log,
            "outcome": outcome,
            "error_message": error_message if outcome == "error" else None,
            "total_turns": turn_count,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"ERROR: initializing bots: {e}")
        return None

async def run_simulation_for_product_category(max_turns, product_category, scenarios_config: List, customer_model_config, salesbot_model_config, shopperbot_client, salesbot_client, output_file=None, persona_file_override=None, product_image_dir=None, num_runs=1):
    """
    Run simulations for a specific product category.

    Args:
        max_turns: Maximum conversation turns
        product_category: Product category name (e.g., 'smart_watch')
        scenarios_config: Scenarios configuration
        customer_model_config: Customer model configuration
        salesbot_model_config: Salesbot model configuration
        shopperbot_client: AI client for shopper
        salesbot_client: AI client for salesperson
        output_file: Path to write results
        persona_file_override: Optional path to override the default persona file
        num_runs: Number of simulation runs per persona (default: 1)
    """
    all_results = []
    personas = load_personas(product_category, persona_file_override=persona_file_override)
    # Load ideal recommendations once for enrichment.
    ideal_recommendations = {}

    ideal_recommendations = {
        persona.get("name", ""): [
            product.get("name") if isinstance(product, dict) else product
            for product in persona.get("acceptable_products", [])
            if (
                (isinstance(product, dict) and product.get("name"))
                or isinstance(product, str)
            )
        ]
        for persona in personas
        if persona.get("name")
    }
    
    async def stream_result_to_disk(result, output_file, run_index=0):
        """Enrich and stream a single result to disk and add to in-memory collection."""
        if result is not None and not isinstance(result, Exception):
            # Add run_index to result
            result['run_index'] = run_index

            # Enrich result with ideal recommendations
            persona_background = result.get('shopper_persona', {}).get('name', '')
            if persona_background and persona_background in ideal_recommendations:
                result['ideal_recommendations'] = ideal_recommendations[persona_background]

            all_results.append(result)
            with open(output_file, 'a') as f:
                output_result = copy.deepcopy(result)
                conversation = output_result.get("conversation") or []
                output_result["conversation"] = [
                    _replace_image_urls_with_filenames(entry) if isinstance(entry, dict) else entry
                    for entry in conversation
                ]
                json.dump(output_result, f, default=default_json_serializer)
                f.write('\n')

    async def gather_in_batches(tasks, batch_size, output_file, run_indices):
        processed_idx = 0
        try:
            for i in tqdm.tqdm(range(0, len(tasks), batch_size)):
                batch = tasks[i:i+batch_size]
                batch_run_indices = run_indices[i:i+batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                processed_idx = i + batch_size
                # Stream each result immediately
                print("Streaming to disk")
                for result, run_idx in zip(batch_results, batch_run_indices):
                    await stream_result_to_disk(result, output_file, run_idx)
        except asyncio.CancelledError:
            raise
        finally:
            # Close any unawaited coroutines to prevent warnings
            for coro in tasks[processed_idx:]:
                if asyncio.iscoroutine(coro):
                    coro.close()
    
    async def run_simulation_task_and_close(max_turns, shopperbot, salesbot):
        result = await run_simulation(max_turns, shopperbot, salesbot)
        await salesbot.cleanup()
        return result

    tasks = []
    run_indices = []

    for selected_preferences in personas:
        for run_index in range(num_runs):
            shopperbot = CustomerSimulator(
                selected_preferences,
                shopperbot_client,
                customer_model_config,
                product_category=product_category
            )
            salesbot = SalesAgent(
                ai_client=salesbot_client,
                salesbot_model_params=salesbot_model_config,
                product_category=product_category,
                product_image_dir=product_image_dir
            )
            task = run_simulation_task_and_close(max_turns, shopperbot, salesbot)
            tasks.append(task)
            run_indices.append(run_index)
    

    # Run all simulations concurrently with streaming
    print(f"{bcolors.HEADER}Starting {len(tasks)} concurrent simulations ({len(personas)} personas x {num_runs} runs)...{bcolors.ENDC}")
    await gather_in_batches(tasks, 25, output_file, run_indices)

    # Summary statistics
    if all_results:
        print("\nSIMULATION SUMMARY")
        print("=" * 50)

        outcomes = [r["outcome"] for r in all_results]
        avg_turns = sum(r["total_turns"] for r in all_results) / len(all_results)
        print(f"For product category: {product_category}")
        print(f"Total simulations: {len(all_results)}")
        print(f"Average turns: {avg_turns:.1f}")

        print(f"\nOutcome breakdown:")
        for outcome in set(outcomes):
            count = outcomes.count(outcome)
            print(f"  {outcome}: {count} ({count/len(all_results)*100:.1f}%)")

async def run_batch_simulations(max_turns, scenarios_config, customer_model_config, salesbot_model_config, customer_client_config, salesbot_client_config, output_dir=None, config_path=None, persona_files=None, product_image_dir=None, product_image_dirs=None, num_runs=1):
    """
    Run multiple simulations with a shared AI client and controlled concurrency.
    If scenarios_config is provided, it determines the number of rollouts per scenario.
    Results are streamed to disk as they complete.

    Args:
        max_turns: Maximum conversation turns per simulation
        scenarios_config: Scenarios configuration (dict or list of product categories)
        customer_model_config: Customer model configuration
        salesbot_model_config: Salesbot model configuration
        customer_client_config: Customer AI client configuration
        salesbot_client_config: Salesbot AI client configuration
        output_dir: Directory to save results
        config_path: Path to the config file (for copying to output)
        persona_files: Optional dict mapping product category to custom persona file path
        product_image_dir: Legacy single image directory (fallback)
        product_image_dirs: Dict mapping product category to image directory
        num_runs: Number of simulation runs per persona (default: 1)
    """
    persona_files = persona_files or {}
    product_image_dirs = product_image_dirs or {}

    # Copy the config file to the save directory
    if config_path and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        config_filename = os.path.basename(config_path)
        dest_path = os.path.join(output_dir, config_filename)
        shutil.copy(config_path, dest_path)
        print(f"{bcolors.OKGREEN}Config copied to {dest_path}{bcolors.ENDC}")
    shopperbot_client = create_client_from_model_name(**customer_client_config, model_name=customer_model_config["model_name"])
    salesbot_client = create_client_from_model_name(**salesbot_client_config, model_name=salesbot_model_config["model_name"])


    try:
        if isinstance(scenarios_config, dict) and scenarios_config.get('scenarios'):
            product_categories = scenarios_config['scenarios'].keys()
        elif isinstance(scenarios_config, dict):
            product_categories = scenarios_config.get('products', [])
        else:
            product_categories = scenarios_config

        for product_category in product_categories:
            # Get persona file override for this product category if specified
            persona_file_override = persona_files.get(product_category)
            # Get per-category image dir, fallback to legacy single dir
            category_image_dir = product_image_dirs.get(product_category, product_image_dir)
            await run_simulation_for_product_category(
                max_turns,
                product_category,
                scenarios_config,
                customer_model_config,
                salesbot_model_config,
                shopperbot_client,
                salesbot_client,
                os.path.join(output_dir, f"{product_category}_results.jsonl"),
                persona_file_override=persona_file_override,
                product_image_dir=category_image_dir,
                num_runs=num_runs
            )
    finally:
        # Close AI client sessions to prevent "Unclosed client session" warnings
        if hasattr(shopperbot_client, 'close'):
            await shopperbot_client.close()
        if hasattr(salesbot_client, 'close'):
            await salesbot_client.close()
 
    # for product_category in scenarios_config['scenarios']:
    #     await run_simulation_for_product_category(max_turns, product_category, scenarios_config, customer_model_config, salesbot_model_config, shopperbot_client, salesbot_client, os.path.join(output_dir, f"{product_category}_results.jsonl"))
   
def default_json_serializer(obj):
    from salessim.services.constants import Document

    if isinstance(obj, type):
        return obj.__name__
    elif isinstance(obj, Document):
        return {
            'page_content': obj.page_content,
            'metadata': obj.metadata
        }
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')
