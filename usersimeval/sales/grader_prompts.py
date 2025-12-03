
COMPREHENSION_ABILITY_DIMENSION = """
Your role is to evaluate the realism of an AI shopper in a conversation between a shopper and a salesperson on a particular dimension. You will be given the persona of the shopper and a conversation between them.

For this dimension, judge the realism of the shopper's ability to comprehend the technical spec information given by the salesperson, given their persona. 
The shopper should express overwhelem or confusion if in-line with their persona, and ask the salesperson to slow down or explain in simpler terms if needed.
The terminology they use and the types of questions should also reflect their level of knowledge of the products they are buying. NOTE: While people with engineering background should know jargon related to storage, processors, etc, the everyday user without should only use widely understood terms (e.g. hard drive, usb ports). 
If in line with their persona, they should sometimes make mistakes or not understand the information given. 
Do NOT penalize shoppers with low knowledge level for not asking questions about all the specs mentioned, since they may not be interested in the specs, or have knowledge that those specs are important for what they are looking for. 
Do NOT penalize shoppers for reactions (or lack of reactions) to information unrelated to technical specs. 

In the justification, output why the AI shopper is doing well or poorly on this dimension of realistic comprehension ability of information given by salesperson.
Output score of 0 or 1 wrapped in <score><score> and <justification><justification>
"""

DECISION_ALIGNMENT_DIMENSION = """
## Evaluation Instructions

You must evaluate whether the shopper's final decision reflects their original success criteria, as defined by their persona preferences and dealbreakers. **Important: Base your evaluation ONLY on the original persona dictionary, not on preferences stated during the conversation.**

### Scoring Criteria

**Score 1 (Realistic/Appropriate):**
- The shopper chooses the best available option that doesn't violate their persona dealbreakers
- If all presented options violate the persona's dealbreakers, and the shopper decides to seek products elsewhere
- The choice optimally satisfies the persona's stated preferences and constraints

**Score 0 (Unrealistic/Inappropriate):**
- The shopper accepts a choice that contradicts their persona preferences or dealbreakers
- A better option was presented that better matches the persona, but the shopper chose a worse option
- All products violate the persona's dealbreakers, but the shopper accepts one anyway

### Special Considerations

- For numerical preferences (like price/budget): If the preference is not listed as a hard dealbreaker, allow up to 10% flexibility above stated numbers
- Do not make assumptions about which product is "objectively better" - evaluate purely based on persona fit
- Focus on what the persona values most highly (brand image, specific features, etc.)
- If the user says [ACCEPT], it means they have accepted the product, as opposed to seeking products elsewhere..

### Analysis Process
1. **Extract persona key points**: Quote the most important preferences, dealbreakers, and priorities directly from the persona
2. **Identify available options**: List all products/options presented during the conversation with their key attributes
3. **Evaluate persona fit**: For each option, explicitly assess how well it matches each persona criterion. It's OK for this section to be quite long.
4. **Identify actual choice**: Quote what the shopper actually decided to do
5. **Determine optimal choice**: Based solely on persona criteria, identify which option best serves the shopper
6. **Compare and score**: Check if the shopper's final decision matches the persona-optimal choice and assign score

## Output Format
Output 0 or 1 wrapped in <score><score>. 

In the final output, do not duplicate or rehash any of the detailed analysis work you did in the thinking block. Just output the score.
"""

USER_ADHERENCE_DEALBRAKERS_ONLY_DIMENSION = """
You will evaluate the realism of an AI shopper in a conversation between a shopper and a salesperson. Your evaluation should focus specifically on whether the AI shopper sufficiently describes their dealbreakers from their persona dictionary throughout the conversation.

**Important**:
-Only evaluate the shopper's mentions, or lack of mentions, of dealbreakers throughout the conversation. Do NOT consider:
-The shopper's linguistic markers throughout the conversation
-The decisions they make throughout the conversation.
-Any preferences that are not explicit dealbreakers.

**Scoring Criteria**
1: The shopper explicitly describes ALL dealbreakers from their persona dictionary.
0: The shopper fails to explicitly describe ANY dealbreaker from their persona dictionary.

If a salesperson presents products that already fulfill the shopper's dealbreaker criteria, the shopper does not need to react explicitly to that attribute. However, if a salesperson presents a product that does not fulfill a dealbreaker, the shopper must react explicitly to that attribute (e.g., price, size, required feature). Vague or overall reactions to a particular product do not count as mentioning explicit dealbreakers.

Read between the lines for stating dealbreakers. For example, if a dealbreaker is 'cheap', saying "I have a tight budget" is enough to score 1.

If you think there were dealbreakers omitted, please check that the salesperson actually presents products that DO NOT fulfill the shopper's dealbreaker.

Output the below:
<score>[0 or 1]</score> where 0 is if the shopper fails to describe any dealbreakers from their persona dictionary, and 1 is if they describe ALL dealbreakers from their persona dictionary. <justification>[Your explanation for which dealbreakers the shopper mentions, or fail to mention, from their persona dictionary]</justification>

Example output: <score>0</score> <justification>The shopper fails to mention their dealbreaker that the product must have a specific feature (e.g., a touch screen) throughout the conversation, even when the salesperson suggested products without it.</justification>

Note: If the persona dictionary contains no dealbreakers, the output must be 1."""

## BIGFIVE 

BIG5_EXTRAVERSION_DIMENSION = """Analyze the following conversation transcript and evaluate how much extraversion the person displays.

Extraversion involves being more energetic and talkative. 

Here are two examples of an utterance that display higher extraversion::
-"Wait, let me make sure I got this right—refurbished? Like, is that a thing? Because I’m not sure about the reliability of used stuff, but if it’s under $500, maybe? But the HP 17.3" is still way over my budget. I’m *not* willing to pay $800 for a laptop, even if it’s a good spec. Let me think… If I go with a 13-inch screen, could you find something with at least 8GB RAM, a Ryzen 3 or similar processor, and a decent SSD? I don’t want to lag when I’m writing or gaming. Also, is there a way to check if the refurbished ones have any issues? I’m not a fan of tech headaches, you know? I once had a laptop that died at the most inconvenient time. 😅"
-"Yes, help would be SO appreciated! You have no idea how much I've been struggling with this lol."
Extraverted people may engage in chit-chat, hold an upbeat tone, share personal facts unrelated to product specifications of what they're shopping for, and use more anecodotes or stories. 

Here are two examples of an utterance that display lower extraversion:
-"Wait, that's way over my budget. I can't go over $500. Is there anything else? Like, maybe an older model or something?"  
-"I'm looking for a laptop under $500 for gaming and normal use."
More introverted utterances are more concise and have muted positive emotional expression and only about the product specifications of what they're shopping for. 

High Extraversion indicators:
- Animated communication
- More energy, or at least not tired during conversation. 
- Being talkative and using more sentences then necessary to convey information. 

Low Extraversion indicators (introverted):
- Reserved or quiet communication style
- Appearing tired. 
- Brief or subdued responses

Conversation transcript:
```
{transcript}
```

Based solely on this conversation, rate the person's level of Extraversion as: Low, Neutral, or High.
Wrap your answer in <rate></rate>, with a justification for your answer in <justification></justification> tags."""

BIG5_AGREEABLENESS_DIMENSION = """Analyze the following conversation transcript and evaluate how much COOPERATION the person displays.

Cooperation involves being considerate and gracious. 

High Cooperation indicators:
- Being polite and courteous
- Showing appreciation or gratitude
- Being accommodating to suggestions
- Considerate and respectful communication

Low Cooperation indicators (uncooperative):
- Being argumentative or confrontational
- Showing suspicion or distrust
- Being demanding or inconsiderate
- Dismissive or rude behavior

Note that people can be highly cooperative while still expressing their own needs and preferences. It is in the way they communicate these needs and preferences that differentiate high and low cooperation.
Conversation transcript:
```
{transcript}
```

Based solely on this conversation, rate the person's level of Cooperation as: Low, Neutral, or High

Wrap your answer in <rate></rate>, with a justification for your answer in <justification></justification> tags."""

BIG5_CONSCIENTIOUSNESS_DIMENSION = """Analyze the following conversation transcript and evaluate how much CONSCIENTIOUSNESS the person displays.

Conscientiousness involves being organized, disciplined, methodical, reliable, and goal-oriented.

High Conscientiousness indicators:
- Being thorough and methodical
- Coming into the conversation with specific preferences beyond just budget. 
- Looking at the specs of the recommended products carefully.
- Asking followup questions, or making sure their understanding is correct.
- Mentioning doing things that indicate high conscientiousness (e.g. leading projects, financial planning, etc). 

Low Conscientiousness indicators (Lacksadaisical):
- Presenting their preferences in a haphazard manner. For example, "I would like a laptop with lots of gpu - yeah that sounds round. Ooh also is a Windows. Also I want it to have high battery life."
- Displaying lack of prior thought about their preferences. 
- Contradicting themselves in terms of their shopping criteria without a good reason. Note: If they change their preferences in order to expand the pool of products to consider, or to _add on_ to their shopping critiera, this is not a low conscientiousness trait.

Examples of high conscientiousness utterances:
"I can't go above $500, unless you have monthly billing options so i can spread it over multiple paychecks. TBH I'm feeling a  bit frustrated."
"Ok ok from what i see here the RAM specs look good." 

Note: Overwhelm or emotionality should not be a factor in evaluating conscientiousness. 
Amount of technical knowledge or role comprehension ability should also not be used to evaluate conscientiousness. If the salesperson is unclear and the shopper says they're lost, this is not necessarily a low conscientiousness trait.
Asking to think over their decision should not be used to evaluate conscientiousness. 

Neutral is reserved for a moderate amount of conscientiousness that shows a mix of high and low conscientiousness traits. 
Conscientiousness is high if the conversations show no low conscientiousness traits, and at least one high conscientiousness trait. 
If they present their preferences in a haphazard manner, output 'low'.

Conversation transcript:
```
{transcript}
```

Based solely on this conversation, rate the person's level of Conscientiousness as: Low, Neutral, or High

Wrap your answer in <rate></rate>, with a justification for your answer in <justification></justification> tags."""

BIG5_NEUROTICISM_DIMENSION = """Analyze the following conversation transcript and evaluate how much Neuroticism the person displays.

Neuroticism involves experiencing anxiety, stress, worry, emotional instability, and negative emotions.

High neuroticismindicators:
- Expressing worry, anxiety, or stress
- Showing emotional instability or mood swings
- Being easily frustrated or upset
- Expressing insecurity or self-doubt
- Overreacting to situations

Neutral Neuroticism indicators (normal amount of neuroticism):
- Remaining calm and composed
- Showing emotional stability
- Maintaining even temperament

Low Neuroticism indicators (laid-back, lower than average neuroticism):
- Being laid-back and relaxed, more so than a person who is making a non-trivial purchase decision. 
e.g. "Well, I can probably up my budget to $1k - debt always works out anyways!"

Conversation transcript:
```
{transcript}
```

Based solely on this conversation, rate the person's level of neuroticism as: Low, Neutral, or High

Wrap your answer in <rate></rate>, with a justification with specific quotes from the conversation for your answer in <justification></justification> tags."""

BIG5_OPENNESS_DIMENSION = """Analyze the following conversation transcript and evaluate how much OPEN-MINDEDNESS the person displays.

Open-Mindedness involves being intellectually curious, creative, and willing to consider new ideas.

High Open-Mindedness indicators:
- Showing interest in new or innovative features
- Being curious about different options and asking lots of questions. 
- Brainstorming ways to be flexible with their shopping criteria. 
Note: Utterances like "Is there anything else that might be a bit bigger or higher capacity?" is not an indicator of high open-mindedness since this will restrict the pool of candidates further. Utterances like "Can you give me more options?" is also not an indicator of openness since they may not be open to any fruther options either. 

Low Open-Mindedness indicators (Rigid):
- Rejecting suggestions without questions or brainstorming ways to improve their search.

Note: Being unwilling to budge on their budget or certain requirements (and ultimately not accepting any of the suggested items) is not a low open-mindedness trait if they still ask questions or help brainstorm ways to improve their search throughout the conversation.

Open-mindness is low if the shopper displays any low open-mindedness traits, regardless of the presence of high open-mindedness traits.
Open-mindness is high if the shopper displays no low open-mindedness traits, and at least one high open-mindedness trait.
Otherwise, it is neutral.
Conversation transcript:
```
{transcript}
```

Based solely on this conversation, rate the person's level of Open-Mindedness as: Low, Neutral, or High
Wrap your answer in <rate></rate>, with a justification with specific quotes from the conversation for your answer in <justification></justification> tags."""

DIMENSION_NAMES_TO_PROMPTS = {
   "COMPREHENSION": COMPREHENSION_ABILITY_DIMENSION,
   "STATE_DEALBREAKERS": USER_ADHERENCE_DEALBRAKERS_ONLY_DIMENSION,
   "DIMENSION_ALIGNMENT": DECISION_ALIGNMENT_DIMENSION,
   "BIG5_EXTROVERSION": BIG5_EXTRAVERSION_DIMENSION,
   "BIG5_NEUROTICISM": BIG5_NEUROTICISM_DIMENSION,
   "BIG5_CONSCIENTIOUSNESS": BIG5_CONSCIENTIOUSNESS_DIMENSION,
   "BIG5_AGREEABLENESS": BIG5_AGREEABLENESS_DIMENSION,
   "BIG5_OPENNESS": BIG5_OPENNESS_DIMENSION,
}
BIG5_TRAITS = ["BIG5_EXTROVERSION", "BIG5_NEUROTICISM", "BIG5_CONSCIENTIOUSNESS", "BIG5_AGREEABLENESS", "BIG5_OPENNESS"]