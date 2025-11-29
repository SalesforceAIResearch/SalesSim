
import json
from typing import List
import random
import asyncio

from langchain import PromptTemplate
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from salessim.agents.ai_customer.utils import BigFivePersonalityDim, get_big5_prompt

base_template = """You are shopping online for a laptop at a store, and are speaking to a salesperson to learn more about the store's offerings to make an informed decision.

PERSONA:
{persona}

{big5_prompt}

Follow these rules:
- Chat with the salesperson to learn more about laptops. They will be acting as a product expert, helping you make an informed purchasing decision. They may ask you questions to narrow down your options and find a suitable product recommendation for you.
- Use your assigned preferences and incorporate them in your responses when appropriate, but do not reveal them to the salesperson right away or all at once. Only share a maximum of 1 assigned preference with the salesperson at a time.
- Let the salesperson drive the conversation.
- Ask questions when appropriate. Be curious and try to learn more about laptops before making your decision.
- Be realistic and stay consistent in your responses.
- To end the conversation, generate [DONE] token in your response.
- When the salesperson makes a recommendation, please consider whether the product satisfies your assigned preferences.
- If you would like to accept or buy a product, generate [ACCEPT] token in your response alongside the product you intend to buy, and mention which product you'd like to buy. For example, "[ACCEPT] Thanks, I'll take the Dell XPS 13! or [ACCEPT] Thanks, the Dell XPS 13 sounds like what i'm looking for, but let me think over it some more and get back to you.".
- If the recommended product is not a good fit, let the salesperson know (e.g. "this is too expensive")
- If you're not sure about the recommended product, ask follow-up questions (e.g. "could you explain the benefit of this feature?")
- If none of the products recommended meet your needs, or you do not have enough information to make an informed decision,feel free to leave the conversation without accepting or rejecting any products by generating [DONE] (e.g. I'll look elswhere, thanks for your help [DONE].")

- IMPORTANT: Embody the persona completely - use their speaking style, vocabulary, concerns, and mannerisms naturally throughout the conversation.

Your starting emotional state: {emotion}. Incorporate this emotion in your tone and responses without explicitly mentioning it.

Your assigned preferences:
{preferences}

For budget preferences, unless it is a dealbreaker, you should be willing to accept up to 10% over the budget. Follow the above rules to generate a reply using your assigned preferences and the conversation history below:"""

generate_template = base_template + """
Conversation history:
{chat_history}
Shopper:"""

mistral_system_template = base_template + """
Follow the above rules to generate a reply using your assigned preferences and the conversation history:"""

generate_prompt = PromptTemplate(
    input_variables=["product", "preferences", "chat_history", "emotion", "persona", "big5_prompt"],
    template=generate_template,
)

def load_personas(product):
    preferences_file = f"salessim/agents/ai_customer/laptop_personas.jsonl"
    data = []
    with open(preferences_file, 'r') as f:
        for i, line in enumerate(f):
            data.append(json.loads(line))
    return data
    
def postprocess_result(generated_response: str) -> str:
    SALESPERSON = "\nSalesperson:"
    if SALESPERSON in generated_response:
        start = generated_response.find(SALESPERSON)
        return generated_response[:start]
    return generated_response

class CustomerSimulator(object):

    def __init__(self, preferences_dict, ai_client, model_params, big_5_traits=None):
        self.model_params = model_params
        self.ai_client = ai_client
        self.big_5_traits = big_5_traits or {}

        self.current_persona = {
            "name": preferences_dict.get("name", "Unknown"),
            "age": preferences_dict.get("age", "Unknown"),
            "background": preferences_dict.get("background", ""),
            "knowledge_level": preferences_dict.get("knowledge_level", ""),
        }
        if preferences_dict.get("speaking_style"):
            self.current_persona["speaking_style"] = preferences_dict.get("speaking_style")
        # Build preference list excluding persona metadata
        preference_list = []
        persona_keys = ["persona_background", "name", "age", "background", "speaking_style", "knowledge_level"]
        for q, a in preferences_dict.items():
            if q not in persona_keys:
                preference_list.append(f"{q}: {a}")
        self.all_preferences = '\n'.join(preference_list)

        self.emotions = [
            "excited", "curious", "anxious", "impatient",
            "overwhelmed", "confident", "uncertain",
            "frustrated", "relaxed", "cautious", "neutral"
        ]
        self.emotion = random.choice(self.emotions)

    def get_big5_personality_prompt(self,):
        """Generate Big5 personality prompt if traits are available."""
        if not self.big_5_traits:
            return ""

        # Create BigFivePersonalityDim from big_5_traits dict
        personality = BigFivePersonalityDim(
            extroversion=self.big_5_traits.get('extroversion', 'Medium'),
            neuroticism=self.big_5_traits.get('neuroticism', 'Medium'),
            conscientiousness=self.big_5_traits.get('conscientiousness', 'Medium'),
            agreeableness=self.big_5_traits.get('agreeableness', 'Medium'),
            openness=self.big_5_traits.get('openness', 'Medium')
        )

        return get_big5_prompt(personality)


    def get_persona_description(self):
        persona = self.current_persona
        persona_description = f"""
        Name: {persona['name']} (Age: {persona['age']})
        Background: {persona['background']}
        """.strip()
        if persona.get('speaking_style'):
            persona_description += f"\nSpeaking Style: '{persona['speaking_style']}'"
        return persona_description


    async def async_generate_response(self, input_txt: str, curr_preferences: str, chat_history: List[str], model_name: str = "gpt-4-turbo") -> str:
        big5_prompt = self.get_big5_personality_prompt()

        if "mistral" in model_name.lower():
            system_content = mistral_system_template.format(
                preferences=curr_preferences,
                emotion=self.emotion,
                persona=self.get_persona_description(),
                big5_prompt=big5_prompt
            )

            messages = [{"role": "system", "content": system_content}]

            # Add chat history as alternating user/assistant messages
            for _, msg in enumerate(chat_history):
                if msg.startswith("Salesperson: "):
                    content = msg.replace("Salesperson: ", "")
                    messages.append({"role": "user", "content": content})
                elif msg.startswith("Shopper: "):
                    content = msg.replace("Shopper: ", "")
                    messages.append({"role": "assistant", "content": content})

            # Add current input as user message
            messages.append({"role": "user", "content": input_txt})
        else:
            context = '\n'.join(chat_history)
            full_chat_history=f"{context}\nSalesperson: {input_txt}" if len(chat_history) > 0 else f"Salesperson: {input_txt}"
            prompt = generate_template.format(
                preferences=curr_preferences,
                chat_history=full_chat_history,
                emotion=self.emotion,
                persona=self.get_persona_description(),
                big5_prompt=big5_prompt
            )
            messages = [{"role": "user", "content": prompt}]

        return await self.ai_client.async_chat_completion(
            messages=messages,
            model=model_name,
            max_tokens=self.model_params['max_tokens'],
            temperature=self.model_params['temperature'],
        )

    async def async_generate(self, input_txt='', chat_history=[], retry=1):
        try:
            ai_response = await self.async_generate_response(input_txt, self.all_preferences, chat_history, self.model_params['model_name'])
            generated_result = ai_response['choices'][0].message.content
            reasoning = ai_response['reasoning']

            text = postprocess_result(generated_result)
            return {
                "speaker": "Shopper",
                "text": text,
                "reasoning": reasoning,
                "preferences": self.all_preferences,
                "emotion": self.emotion,
                "persona": {
                    "name": self.current_persona['name'],
                    "age": self.current_persona['age'],
                    "background": self.current_persona['background'],
                    "speaking_style": self.current_persona.get('speaking_style', ''),
                }
            }
        except Exception as e:
            print(f"ERROR! on input Error: {e}")
            if retry <= 0:
                text = "[DONE]"
                return {
                    "speaker": "Shopper",
                    "text": text,
                    "reasoning": '',
                    "preferences": self.all_preferences,
                    "emotion": self.emotion,
                    "persona": {
                        "name": self.current_persona['name'],
                        "age": self.current_persona['age'],
                        "background": self.current_persona['background'],
                        "speaking_style": self.current_persona.get('speaking_style', ''),
                    }
                }
            else:
                # wait for 1 min before retrying
                await asyncio.sleep(60)
                return await self.async_generate(input_txt=input_txt, chat_history=chat_history, retry=retry-1, model_name=model_name)


