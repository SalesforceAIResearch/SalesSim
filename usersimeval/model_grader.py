#!/usr/bin/env python3
"""
Model grader using gpt-4o and claude-sonnet-4-5, the models used during grader development.
"""

import json
import os
import asyncio
from typing import List, Dict, Any
from pathlib import Path
import logging
from datetime import datetime
from usersimeval.convert_rollouts_to_txt import convert_conversations_to_txt
from usersimeval.utils import aggregate_big5_scores, aggregate_float_scores, get_big5_scores, get_mode_score, extract_scores

from usersimeval.sales.grader_prompts import *
try:
    from common.ai_client import LiteLLMClient
except ImportError:
    print("Please install required packages: pip install litellm")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_input_file(input_file: str) -> List[Dict]:
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            # Handle both single objects and arrays
            if isinstance(data, list):
                conversations = data
            else:
                conversations = [data]
            return conversations
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {e}")
            raise


class UserSimulatorJudge:
    def __init__(self, num_tries_per_conversation: int = 10):
        """
        Initialize the LLM judge using ai_client

        Args:
            num_tries_per_conversation: Number of tries per conversation
        """
        self.num_tries_per_conversation = num_tries_per_conversation
        # Use ai_client for all model interactions
        self.openai_client = LiteLLMClient(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.anthropic_client = LiteLLMClient(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )


    def _format_conversation(self, conversation: List[Dict]) -> str:
        """Format conversation for the prompt"""
        formatted = []
        for turn in conversation:
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '').replace("[ACCEPT]", "").replace("[REJECT]", "")
            text = text.replace("[DONE]", "")
            formatted.append(f"{speaker}: {text}")

        return "\n".join(formatted)

    async def preprocess_big5_prompt(self, conversation_history: List[Dict]) -> str:
        shopper_utterances = []
        for turn in conversation_history:
            if turn['speaker'] == 'Shopper':
                shopper_utterances.append(turn['text'])

        transcript = "\nUtterance: ".join(shopper_utterances)
        return transcript

    async def get_feedback_for_dimension(self, dimension_name, conversation_history, persona, formatted_conversation):
        for attempt in range(3):
            try:
                if dimension_name in BIG5_TRAITS:
                    transcript = await self.preprocess_big5_prompt( conversation_history)
                    prompt = DIMENSION_NAMES_TO_PROMPTS[dimension_name].format(transcript=transcript)
                    response = await self.openai_client.async_chat_completion(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model="gpt-4o",
                        max_tokens=1000,
                        temperature=0.3
                    )
                    return response['choices'][0].message.content.strip()
                else:
                    prompt = DIMENSION_NAMES_TO_PROMPTS[dimension_name]
                    user_content = f"Here is the persona of the shopper and the conversation to evaluate:\nPersona:\n{persona}\nConversation:\n{formatted_conversation}"
                    response = await self.anthropic_client.async_chat_completion(
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": user_content}
                        ],
                        model="claude-sonnet-4-5",
                        max_tokens=4000,
                        temperature=0.3
                    )
                    # Format response with reasoning if available
                    if response['reasoning']:
                        return f"{response['choices'][0].message.content}<justification>{response['reasoning']}</justification>"
                    else:
                        return response['choices'][0].message.content
            except Exception as e:
                print("Retrying")
                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    raise
                    
    async def judge_conversation(self, conversation_data: Dict[str, Any], dimensions: List[str]) -> Dict[str, Any]:
        """
        Judge a single conversation and return feedback
        """
        try:
            conversation_history = eval(conversation_data.get('conversation', [])) if type(conversation_data.get('conversation', [])) is str else conversation_data.get('conversation', [])
            persona = eval(conversation_data.get('shopper_persona', '')) if type(conversation_data.get('shopper_persona', '')) is str else conversation_data.get('shopper_persona', '')
            preferences = conversation_data.get('shopper_preferences', '')
            emotion = conversation_data.get('shopper_emotion', '')
            persona.update({'preferences': preferences, 'emotion': emotion})
            formatted_conversation = self._format_conversation(conversation_history)
            dimension_results = {}

            # Process each dimension
            for _, dimension_name in enumerate(dimensions, 1):
                    feedbacks = await asyncio.gather(
                        *[self.get_feedback_for_dimension(dimension_name, conversation_history, persona, formatted_conversation) for _ in range(self.num_tries_per_conversation)]
                    )
                    dimension_results[dimension_name] = {
                        "status": "success",
                        "feedback": feedbacks,
                    }
        except Exception as e:
            logger.error(f"Error judging conversation: {e}")
            return {
                "status": "error",
                "error": str(e),
                "raw_response": ""
            }
        return dimension_results

    async def process_json_file(self, input_file: str, output_dir: str, dimensions: List[str] = []) -> List[Dict]:
        """
        Process all conversations in the JSON file and write results incrementally
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        output_file = os.path.join(output_dir, f"breakdown_scores.json")
        results = []
        conversations = load_input_file(input_file)
        # Initialize output file with empty array
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump([], out_f, indent=2, ensure_ascii=False)
        
        # Process each conversation
        skipped_dialogues = []
        for idx, conversation_data in enumerate(conversations):
            try:
                # Get conversation ID (fallback to index for backward compatibility)
                conversation_id = conversation_data.get('conversation_id', str(idx))

                # Skip conversations with "error" outcome
                outcome = conversation_data.get('outcome', '')
                if outcome == 'error':
                    skipped_dialogues.append(conversation_id)
                    continue

                logger.info(f"Processing conversation {conversation_id}...")

                judgment = await self.judge_conversation(conversation_data, dimensions)
                # Process results for each dimension
                dimension_scores = {}
                judgment_verbose = {}
                for dim_name, feedbacks in judgment.items():
                    if dim_name in BIG5_TRAITS:
                        votes = []
                        justifications = []
                        scores = feedbacks["feedback"]
                        for score in scores:
                            vote = score.split("<rate>")[1].split("</rate>")[0]
                            justification = score.split("<justification>")[1].split("</justification>")[0]
                            votes.append(vote)
                            justifications.append(justification)
                        mode_score = get_big5_scores(votes, dim_name)
                    else:
                        scores = extract_scores(feedbacks["feedback"])
                        mode_score = get_mode_score(scores)
                    dimension_scores[dim_name] = mode_score
                    judgment_verbose[dim_name] = feedbacks['feedback']
                result = {
                    "conversation_id": conversation_id,
                    "dimension_scores": dimension_scores,
                    "judgment_verbose": judgment_verbose,
                }

                results.append(result)

                # Write results incrementally to file
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    json.dump(results, out_f, indent=2, ensure_ascii=False)
                
                logger.info(f"Results for conversation {conversation_id} written to {output_file}")

            except Exception as e:
                logger.error(f"Error processing conversation {conversation_id}: {e}")
                continue

        logger.info(f"Processing complete. All results saved to: {output_file}")
        if skipped_dialogues:
            logger.info(f"Skipped {len(skipped_dialogues)} dialogues with error outcomes: {skipped_dialogues}")

        # Add skip information to results for aggregate reporting
        skip_info = {
            "skipped_count": len(skipped_dialogues),
            "skipped_conversation_ids": skipped_dialogues
        }

        return results, skip_info
    
def write_aggregate_scores(results, dimensions, input_file, output_dir, skip_info=None, output_filename="aggregate_eval_scores.json"):
    """
    Calculate the average (mean) score per dimension across all conversations in results,
    and write the result to output_dir/output_filename as JSON.
    """
    import json
    conversations = load_input_file(input_file)
    final_output = {}
    for dimension in dimensions:
        if dimension in BIG5_TRAITS:
            scores = aggregate_big5_scores(results, dimension, conversations)
        else:
            scores = aggregate_float_scores(results, dimension)
        final_output[f"{dimension}_SCORE"] = scores

    if skip_info:
        final_output["skipped_dialogues"] = skip_info

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

async def main():
    """Main function to run the LLM judge"""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Judge using Anthropic Claude via GCP Vertex AI")
    parser.add_argument("--input_file", required=True, default="laptop-sales-manual-annotation.json", help="Input JSON file")
    parser.add_argument("--output_dir", help="Output file (auto-generated if not provided)")
    parser.add_argument("--dimensions", required=True, nargs="+", help="Dimensions to evaluate")
    parser.add_argument("--num_tries_per_conversation", type=int, default=5, help="Number of tries per conversation")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f"judge_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.output_dir, exist_ok=True)
    convert_conversations_to_txt(args.input_file, os.path.join(args.output_dir, "human_readable_conversations"))
    judge = UserSimulatorJudge(
        num_tries_per_conversation=args.num_tries_per_conversation
    )
    if args.dimensions:
        if args.dimensions == ["ALL"]:
            args.dimensions = list(DIMENSION_NAMES_TO_PROMPTS.keys())
        else:
            for dimension in args.dimensions:
                if dimension not in DIMENSION_NAMES_TO_PROMPTS:
                    print(f"Dimension {dimension} not found in DIMENSION_NAMES_TO_PROMPTS")
                    raise ValueError(f"Dimension {dimension} not found in DIMENSION_NAMES_TO_PROMPTS")
    results, skip_info = await judge.process_json_file(args.input_file, args.output_dir, args.dimensions)
    write_aggregate_scores(results, args.dimensions, args.input_file, args.output_dir, skip_info)
    
    print(f"Successfully processed {len(results)} conversations")
if __name__ == "__main__":
    asyncio.run(main())