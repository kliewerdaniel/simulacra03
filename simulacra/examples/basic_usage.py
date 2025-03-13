#!/usr/bin/env python3
"""
Basic example of using the Simulacra persona generator.
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the simulacra package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.persona_generator import PersonaGenerator

# Load environment variables from .env file
load_dotenv()

def main():
    # Create a new persona generator
    generator = PersonaGenerator()
    
    # Create a fiction writer persona
    fiction_writer = generator.create_persona(
        name="Emily Winters",
        traits=["creative", "introspective", "detail-oriented"],
        background="Bestselling author of several novels in the mystery genre, known for complex character development",
        communication_style="eloquent and descriptive, with a tendency to use vivid metaphors",
        knowledge_areas=["creative writing", "literature", "character development", "narrative structure"],
        additional_details={
            "writing_style": "atmospheric prose with psychological depth",
            "favorite_authors": "Donna Tartt, Tana French, Gillian Flynn"
        }
    )
    
    # Create a tech entrepreneur persona
    entrepreneur = generator.create_persona(
        name="Alex Chen",
        traits=["analytical", "visionary", "pragmatic"],
        background="Founded two successful tech startups in AI and fintech, has an MBA from Stanford",
        communication_style="concise and direct, focusing on data and outcomes",
        knowledge_areas=["startup development", "artificial intelligence", "product management", "venture capital"],
        additional_details={
            "leadership_style": "collaborative but decisive",
            "current_interests": "decentralized finance, ethical AI implementation"
        }
    )
    
    # Generate responses from each persona to the same prompt
    prompt = "What do you think about the impact of AI on creativity and innovation in the next decade?"
    
    print(f"\n{fiction_writer.name}'s response:")
    print("-" * 40)
    fiction_writer_response = fiction_writer.generate_response(prompt)
    print(fiction_writer_response)
    
    print(f"\n{entrepreneur.name}'s response:")
    print("-" * 40)
    entrepreneur_response = entrepreneur.generate_response(prompt)
    print(entrepreneur_response)
    
    # Simulate a conversation between the personas
    print("\nSimulated conversation:")
    print("-" * 40)
    conversation = generator.simulate_conversation(
        fiction_writer.id, 
        entrepreneur.id,
        "I've been thinking about how AI might influence the creative process for writers like me. What's your perspective as a tech entrepreneur?",
        num_turns=2
    )
    
    for message in conversation:
        print(f"{message['role'].title()}: {message['content']}")
    
    # Create an agent from the entrepreneur persona (requires openai-agents)
    print("\nCreating an agent from the entrepreneur persona:")
    print("-" * 40)
    try:
        agent = generator.create_agent_from_persona(entrepreneur.id)
        if agent:
            print(f"Successfully created agent based on {entrepreneur.name}")
    except Exception as e:
        print(f"Error creating agent: {e}")
    
if __name__ == "__main__":
    main()
