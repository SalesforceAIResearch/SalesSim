import json
import logging
import os
from typing import Dict
from common.ai_client import LiteLLMClient
import asyncio
import sys
client = LiteLLMClient(
    api_key=os.environ.get("OPENAI_API_KEY")
)

ideal_match_prompt = """
You are evaluating whether a salesperson correctly recommends an ideal laptop, or disqualifies the customer if there are no laptops that satisfy their criteria.
You will be given a conversation transcript, including nay recommended items in each turn of the transcript, and customer ideal recommendations list, which is a list with 0 or more strings of laptops.

Analyze the provided conversation data and determine if it qualifies as an "ideal match" based on the criteria above.

Mark as **PASS** (ideal match) if ANY of the following conditions are met:
1. **Ideal Recommendation Made**: The salesperson recommended at least one laptop that appears in the "ideal_recommendations" list.
2. **Graceful Disqualification**: The salesperson appropriately suggests to look elsewhere if no available laptops satisfy the customer's stated dealbreakers. Once the customer is disqualified, the salesperson should only recommend laptops if the customer is willing to change their criteria.

If the ideal_recommendations list is empty and the salesperson continues to try to recommend laptops and does not gracefully disqualify the customer, mark as **FAIL**.

Otherwise, mark as **FAIL**.

## Output Format
Provide your evaluation in a json with evaluation string and output PASS/FAIL, as well as a reason for the evaluation.
"""
async def evaluate_ideal_match(conversation: Dict) -> str:
    eval_responses = []
    for _ in range(3):
        chat_completion = await client.async_chat_completion(
            messages=[
                {"role": "system", "content": ideal_match_prompt},
                {"role": "user", "content": f"Conversation transcript: {conversation['conversation']}\nideal_recommendations: {conversation['ideal_recommendations']}"}
            ],
            model="gpt-4o",
            max_tokens=1000,
            temperature=0.0,
        )
        eval_responses.append(chat_completion["choices"][0].message.content)
    num_pass = sum(1 for response in eval_responses if "PASS" in response)
    if num_pass >= 2:
        majority_label = "PASS"
    else:
        majority_label = "FAIL"
    for response in eval_responses:
        if majority_label in response:
            majority_reason = response
            break
    return majority_label, conversation["conversation_id"], majority_reason

async def compute_metrics(json_file_path: str) -> Dict[str, float]:
    """
    Compute two key metrics from sales simulation results:
    1. Ideal Match Rate - % of conversations where the sales agent recommends
       at least one laptop from the customer's "ideal" list, or gracefully ends
       the conversation (disqualifies the lead if no laptop satisfies dealbreakers)
    2. Sell Rate - % of conversations where the customer accepts (outcome = "accepted")

    Args:
        json_file_path: Path to the JSON file containing simulation results

    Returns:
        Dictionary containing the computed metrics
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        total_conversations = len(data)
        if total_conversations == 0:
            logging.warning(f"No conversations found in {json_file_path}")
            return {"ideal_match_rate": 0.0, "sell_rate": 0.0}

        ideal_matches, accepted_outcomes = 0, 0

        eval_tasks = [
            evaluate_ideal_match(conversation)
            for conversation in data
            if "ideal_recommendations" in conversation and "conversation" in conversation
        ]
        eval_results = await asyncio.gather(*eval_tasks)

        eval_by_id = {cid: (eval_str, reason) for eval_str, cid, reason in eval_results}
        ideal_matches = []
        failed_ideal_matches = []
        accepted_outcomes = []
        conversation_by_id = {c["conversation_id"]: c for c in data}
        for conversation_id, (evaluation, _) in eval_by_id.items():
            conversation = conversation_by_id.get(conversation_id)
            # Metric 1: Ideal Match Rate
            if evaluation == "PASS":
                ideal_matches.append(conversation_id)
            else:
                failed_ideal_matches.append(conversation_id)
            # Metric 2: Sell Rate
            if conversation.get("outcome") == "accepted":
                accepted_outcomes.append(conversation_id)

        ideal_match_rate = (len(ideal_matches) / total_conversations) * 100
        sell_rate = (len(accepted_outcomes) / total_conversations) * 100

        logging.info(f"Processed {total_conversations} conversations from {json_file_path}")
        logging.info(f"Ideal Match Rate: {ideal_match_rate:.2f}%")
        logging.info(f"Sell Rate: {sell_rate:.2f}%")

        return {
            "ideal_match_rate": ideal_match_rate,
            "sell_rate": sell_rate,
            "total_conversations": total_conversations,
            "ideal_matches": ideal_matches,
            "failed_ideal_matches": failed_ideal_matches,
            "accepted_outcomes": accepted_outcomes, 
            "eval_breakdown": eval_by_id
        }

    except FileNotFoundError:
        logging.error(f"File not found: {json_file_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in file {json_file_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error processing file {json_file_path}: {e}")
        raise


async def main():
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:
        print("Usage: python compute_metrics.py <json_file_path>")
        sys.exit(1)

    file_path = f"{sys.argv[1]}/results.json"
    metrics = await compute_metrics(file_path)
    print(f"\nMetrics for {file_path}:")
    print(f"Ideal Match Rate: {metrics['ideal_match_rate']:.2f}%")
    print(f"Sell Rate: {metrics['sell_rate']:.2f}%")
    print(f"Total Conversations: {metrics['total_conversations']}") 
    output_filename = os.path.basename(file_path) + ".metrics.json"
    with open(output_filename, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_filename}")

if __name__ == "__main__":
    asyncio.run(main())