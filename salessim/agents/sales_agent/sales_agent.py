import os
import json

from typing import List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.ai_client import AIClient

from agents.sales_agent.prompts import (
    system_instruction,
)
from services.http_clients import ProductLookupClient, BuyingGuideClient
from common.bcolors import bcolors

def postprocess_result(generated_response: str) -> str:
    NEXT_SPEAKER = "\nShopper:"
    if NEXT_SPEAKER in generated_response:
        start = generated_response.find(NEXT_SPEAKER)
        return generated_response[:start]
    return generated_response



class SalesAgent(object):

    def __init__(self, ai_client: AIClient = None, salesbot_model_params: dict = None):
        self.model_params = salesbot_model_params
        self.ai_client = ai_client
        self.buying_guide_client = BuyingGuideClient()
        self.product_catalog_client = ProductLookupClient()
        self.sim_threshold = 0.70
        # Define tool schemas for OpenAI function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_buying_guide",
                    "description": "Search for buying guides and articles about product categories to help understand product features and comparisons",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for buying guide information"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup_product_items",
                    "description": "Search for specific products in the store inventory based on customer requirements",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for product items"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    async def cleanup(self):
        """Clean up HTTP client sessions"""
        await self.buying_guide_client.close()
        await self.product_catalog_client.close()

    def parse_chat_history(self, chat_history):
        messages = []
        for u in chat_history:
            if u.startswith("Shopper: "):
                text = u.replace("Shopper: ", "")
                messages.append({"role": "user", "content": text})
            elif u.startswith("Salesperson: "):
                text = u.replace("Salesperson: ", "")
                messages.append({"role": "assistant", "content": text})
        return messages

    def _build_messages(self, input_txt: str, chat_history: List[str]) -> List[dict]:
        """Build standardized message format for all models"""
        messages = [{"role": "system", "content": system_instruction}]
        messages.extend(self.parse_chat_history(chat_history))
        messages.append({"role": "user", "content": input_txt})
        return messages

    async def _execute_tool_call_async(self, function_name: str, function_args: dict, knowledge_used: list, all_product_candidates: list) -> str:
        """Execute tool calls asynchronously using HTTP clients"""
        print(f"{bcolors.OKGREEN}Action: {function_name}{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Query: {function_args.get('query', function_args.get('message', ''))}{bcolors.ENDC}")

        if function_name == "lookup_buying_guide":
            query = function_args["query"]
            knowledge_candidates = await self.buying_guide_client.top_docs(query, k=4)
            knowledge = "\n---\n".join([item.page_content for item in knowledge_candidates])
            knowledge_used.append({"action": function_name, "query": query, "knowledge": knowledge})

            print(f"{bcolors.OKBLUE}Knowledge: {knowledge}{bcolors.ENDC}")
            return f"Buying guide information:\n{knowledge}"

        elif function_name == "lookup_product_items":
            query = function_args["query"]
            product_candidates = await self.product_catalog_client.top_docs(query, k=4)
            all_product_candidates.extend(product_candidates)
            knowledge = "\n---\n".join([item.page_content for item in product_candidates])
            knowledge_used.append({"action": function_name, "query": query, "knowledge": knowledge})

            print(f"{bcolors.OKBLUE}Knowledge: {knowledge}{bcolors.ENDC}")
            return f"Product information:\n{knowledge}"

        return ""

    async def _format_final_response(self, result: str, reasoning: str, knowledge_used: list, all_product_candidates: list) -> dict:
        """Format the final response in a standardized way"""
        recommended_items = await self.product_catalog_client.find_recommended_items_in_response(
            all_product_candidates, result, self.sim_threshold
        ) if all_product_candidates else []

        all_knowledge = "\n---\n".join([k["knowledge"] for k in knowledge_used])
        result = postprocess_result(result)

        return {
            "speaker": "Salesperson",
            "text": result,
            "reasoning": reasoning,
            "knowledge": all_knowledge,
            "recommended_items": recommended_items,
            "knowledge_used": knowledge_used
        }

    async def async_generate(self, input_txt: str, chat_history: List[str]):
        messages = self._build_messages(input_txt, chat_history)
        knowledge_used = []
        all_product_candidates = []

        # Loop until we get a communicate action
        max_iterations = 3
        iterations = 0
        while True:
            if iterations > max_iterations:
                raise Exception("Max iterations reached, model doesn't want to communicate, just call tools.")
            iterations += 1
            response = await self.ai_client.async_chat_completion(
                messages=messages,
                model=self.model_params['model_name'],
                max_tokens=self.model_params['max_tokens'],
                temperature=self.model_params['temperature'],
                tools=self.tools,
                tool_choice="auto"
            )

            choice = response['choices'][0]
            reasoning = response.get('reasoning', '')

            # If there are tool calls, execute them
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                messages.append(choice.message)

                for tool_call in choice.message.tool_calls:
                    function_name = tool_call.function.name
                    if function_name == "communicate" and self.model_params['model_name'].startswith("gpt-"):
                        function_args = {"message": tool_call.function.arguments}
                    else:
                        function_args = json.loads(tool_call.function.arguments)

                    knowledge_content = await self._execute_tool_call_async(function_name, function_args, knowledge_used, all_product_candidates)

                    if knowledge_content:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": knowledge_content
                        })
            else:
                # No tool calls, treat as direct communication
                return await self._format_final_response(choice.message.content, reasoning, knowledge_used, all_product_candidates)
