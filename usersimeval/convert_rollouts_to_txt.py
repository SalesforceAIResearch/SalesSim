#!/usr/bin/env python3
import json
import os
from pathlib import Path

def convert_conversations_to_txt(json_file_path, output_dir="mistral_conversations_comprehension"):
    """Convert JSON conversations to individual human-readable text files."""

    # Create output directory if it doesn't exist
    if os.path.exists(output_dir):
        # Remove the directory and all its contents if it exist
        import shutil
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(exist_ok=True)

    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each conversation
    if "simulations" in data:
        data = data["simulations"]
    for i, conversation_data in enumerate(data):
        # Extract persona and preferences
        persona = conversation_data.get('shopper_persona', {})
        preferences = conversation_data.get('shopper_preferences', {})
        emotion = conversation_data.get('shopper_emotion', 'neutral')

        # Create filename
        conversation_id = conversation_data.get('conversation_id', i + 1)
        filename = f"conversation_{conversation_id}.txt"
        filepath = Path(output_dir) / filename

        # Generate the text content
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header with persona information
            f.write("=" * 80 + "\n")
            f.write(f"CONVERSATION #{i+1}\n")
            f.write("=" * 80 + "\n\n")

            # Persona details
            f.write("SHOPPER PERSONA:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Name: {persona.get('name', 'Unknown')}\n")
            f.write(f"Age: {persona.get('age', 'Unknown')}\n")
            f.write(f"Background: {persona.get('background', 'Not specified')}\n")
            f.write(f"Personality: {persona.get('shopper_big5_traits', 'Not specified')}\n")
            f.write(f"Speaking Style: {persona.get('speaking_style', 'Not specified')}\n")
            f.write(f"Current Emotion: {emotion}\n")
            #f.write(f"Big5 Traits: {big5_traits}\n")
            if persona.get('concerns'):
                f.write(f"Concerns: {', '.join(persona['concerns'])}\n")

            if persona.get('knowledge_level'):
                f.write(f"Knowledge Level: {persona['knowledge_level']}\n")

            f.write("\nSHOPPER PREFERENCES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Preferences: {preferences}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("CONVERSATION\n")
            f.write("=" * 80 + "\n\n")

            # Write the conversation
            conversation = conversation_data.get('conversation', [])
            for turn in conversation:
                speaker = turn.get('speaker', 'Unknown')
                text = turn.get('text', '')
                turn_num = turn.get('turn', 0)

                f.write(f"[Turn {turn_num}] {speaker.upper()}:\n")
                f.write(f"{text}\n\n")

            f.write("=" * 80 + "\n")
            f.write("END OF CONVERSATION\n")
            f.write("=" * 80 + "\n")

    print(f"Converted {len(data)} conversations to text files in '{output_dir}' directory")
