#!/usr/bin/env python3
import asyncio
import json
import yaml
import asyncio
import traceback
import uuid
from itertools import product
from datetime import datetime
import tqdm
from salessim.agents.sales_agent.sales_agent import SalesAgent
from salessim.agents.ai_customer.ai_customer import load_personas, CustomerSimulator
from common.ai_client import create_client_from_model_name
from common.bcolors import bcolors



def load_scenarios_from_yaml(yaml_file):
    """Load scenarios configuration from YAML file."""
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def generate_scenario_combinations(scenario_config):
    """Generate all unique combinations for a scenario based on Big 5 filters."""
    big_5_specifications = scenario_config.get('big_5_specification', {})

    if not big_5_specifications:
        print(f"{bcolors.WARNING}WARNING: Big 5 specifications are empty. Check your scenarios config.")
        return [scenario_config]

    # Get all trait combinations
    trait_keys = list(big_5_specifications.keys())
    trait_values = []

    for trait in trait_keys:
        values = big_5_specifications[trait]
        if isinstance(values, str):
            trait_values.append([v.strip() for v in values.split(',')])
        else:
            trait_values.append(values)

    # Generate all combinations
    combinations = list(product(*trait_values))

    scenarios = []
    for combo in combinations:
        scenario = scenario_config.copy()
        scenario['big_5_specification'] = dict(zip(trait_keys, combo))
        scenarios.append(scenario)

    return scenarios


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

        # Start with salesbot greeting
        salesperson_text = "Hello! I'm here to help you find the perfect product. What are you looking for today?"

        conversation_log.append({
            "speaker": "Salesperson",
            "text": salesperson_text,
            "turn": turn_count
        })

        # Run conversation simulation
        while turn_count < max_turns:
            try:
                turn_count += 1

                # Shopperbot responds
                shopper_response = await shopperbot.async_generate(
                    input_txt=salesperson_text,
                    chat_history=chat_history,
                )

                shopper_text = shopper_response["text"]
                chat_history.append(f"Salesperson: {salesperson_text}")

                if verbose:
                    print(f"{bcolors.OKCYAN}Shopper: {shopper_text}{bcolors.ENDC}")
                conversation_log.append({
                    "speaker": "Shopper",
                    "text": shopper_text,
                    "reasoning": shopper_response.get("reasoning", ""),
                    "turn": turn_count,
                    "preferences_used": shopper_response.get("preferences", "")
                })


                # Check if shopper accepted a recommendation
                if "[ACCEPT]" in shopper_text:
                    outcome = "accepted"
                    if verbose:
                        print(f"{bcolors.OKGREEN}Recommendation accepted! Simulation complete.{bcolors.ENDC}")
                    break

                if "[DONE]" in shopper_text:
                    outcome = "ended_by_shopper"
                    if verbose:
                        print(f"{bcolors.WARNING}Shopper ended the conversation.{bcolors.ENDC}")
                    break

                # Salesagent responds
                sales_response = await salesbot.async_generate(shopper_text, chat_history)
                chat_history.append(f"Shopper: {shopper_text}")
                salesperson_text = sales_response["text"]

                conversation_log.append({
                    "speaker": "Salesperson",
                    "text": salesperson_text,
                    "turn": turn_count,
                    "reasoning": sales_response.get("reasoning", ""),
                    "knowledge_used": sales_response.get("knowledge", ""),
                    "recommended_items": sales_response.get("recommended_items", []),
                    "recommended_items_count": len(sales_response.get("recommended_items", []))
                })



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

        return {
            "conversation_id": conversation_id,
            "shopper_preferences": shopperbot.all_preferences,
            "shopper_big5_traits": shopperbot.big_5_traits,
            "shopper_persona": shopperbot.current_persona,
            "shopper_emotion": shopperbot.emotion,
            "conversation": conversation_log,
            "outcome": outcome,
            "error_message": error_message if outcome == "error" else None,
            "total_turns": turn_count,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"ERROR: initializing bots: {e}")
        return None



async def run_batch_simulations(max_turns, scenarios_config, customer_model_config, salesbot_model_config, customer_client_config, salesbot_client_config):
    """
    Run multiple simulations with a shared AI client and controlled concurrency.
    If scenarios_config is provided, it determines the number of rollouts per scenario.
    Results are saved incrementally as they complete.
    """

    shopperbot_client = create_client_from_model_name(**customer_client_config)
    salesbot_client = create_client_from_model_name(**salesbot_client_config)


    personas = load_personas("laptop")
    tasks = []
    async def gather_in_batches(tasks, batch_size):
        results = []
        for i in tqdm.tqdm(range(0, len(tasks), batch_size)):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)
        return results
    
    async def run_simulation_task_and_close(max_turns, shopperbot, salesbot):
        result = await run_simulation(max_turns, shopperbot, salesbot)
        await salesbot.cleanup()
        return result

    # Use scenarios configuration to determine simulations
    for scenario_config in scenarios_config['scenarios']:
        # Generate all unique combinations for this scenario
        unique_scenarios = generate_scenario_combinations(scenario_config)
        num_rollouts = scenario_config.get('num_rollouts_per_unique_scenario', 1)

        for unique_scenario in unique_scenarios:
            for _ in range(num_rollouts):
                # Enrich unique_scenario with persona
                selected_preferences = [p for p in personas if p["persona_background"] == unique_scenario["persona"]][0]
                big_5_specifications = unique_scenario.get('big_5_specification', {})
                shopperbot = CustomerSimulator(selected_preferences, shopperbot_client, customer_model_config, big_5_traits=big_5_specifications)
                salesbot = SalesAgent(
                    ai_client=salesbot_client,
                    salesbot_model_params=salesbot_model_config
                )
                task = run_simulation_task_and_close(max_turns, shopperbot, salesbot)
                tasks.append(task)
    # Run all simulations concurrently
    print(f"{bcolors.HEADER}Starting {len(tasks)} concurrent simulations...{bcolors.ENDC}")
    results = await gather_in_batches(tasks, 5)

    # Filter out exceptions and None results
    all_results = [r for r in results if r is not None and not isinstance(r, Exception)]

    # Summary statistics
    if all_results:
        print("\nSIMULATION SUMMARY")
        print("=" * 50)

        outcomes = [r["outcome"] for r in all_results]
        avg_turns = sum(r["total_turns"] for r in all_results) / len(all_results)

        print(f"Total simulations: {len(all_results)}")
        print(f"Success rate: {outcomes.count('accepted')}/{len(all_results)} ({outcomes.count('accepted')/len(all_results)*100:.1f}%)")
        print(f"Average turns: {avg_turns:.1f}")

        print(f"\nOutcome breakdown:")
        for outcome in set(outcomes):
            count = outcomes.count(outcome)
            print(f"  {outcome}: {count} ({count/len(all_results)*100:.1f}%)")
    return all_results


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

def enrich_results_with_ideal_recommendations(results):
    """Enriches the results with ideal recommendations from reference_ideal_recommendations.json."""

    with open('reference_ideal_recommendations.json', 'r') as f:
        ideal_recommendations = json.load(f)

    for result in results:
        persona_background = result.get('shopper_persona').get('name', '')
        if persona_background and persona_background in ideal_recommendations:
            result['ideal_recommendations'] = ideal_recommendations[persona_background]
    return results


def save_results(results, filename=None):
    """Save simulation results to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_results_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=default_json_serializer)

    print(f"{bcolors.OKGREEN}Results saved to {filename}{bcolors.ENDC}")