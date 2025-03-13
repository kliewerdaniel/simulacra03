"""
Style Replication Example

This example demonstrates how to use the StyleReplicationAgent to generate content
in an author's style based on a persona and content brief.
"""

import os
import json
from pathlib import Path
import argparse

# Import necessary components
from src.style_replication.style_replication_agent import (
    StyleReplicationAgent, 
    ContentBrief, 
    StyleParameters,
    GenerationFeedback
)
from src.persona_generator.persona import Persona
from src.agent_workflow import AgentWorkflow


def load_persona(persona_path):
    """Load a persona from a JSON file."""
    with open(persona_path, 'r', encoding='utf-8') as f:
        persona_data = json.load(f)
    
    # Try to determine if this is an AuthorPersona or a Simulacra Persona
    if 'traits' in persona_data and 'background' in persona_data:
        # This looks like a Simulacra Persona
        return Persona.from_dict(persona_data)
    else:
        # Return as a dictionary for the StyleReplicationAgent to handle
        return persona_data


def generate_content(persona_path, topic, content_type, output_dir):
    """
    Generate content in the author's style based on a persona and content brief.
    
    Args:
        persona_path: Path to the persona JSON file
        topic: The topic for the content
        content_type: Type of content to generate (e.g., 'blog post', 'article')
        output_dir: Directory to save the output
    """
    # Load the persona
    print(f"Loading persona from {persona_path}")
    persona = load_persona(persona_path)
    
    # Print persona details
    if hasattr(persona, 'name'):
        print(f"Generating content in the style of: {persona.name}")
    elif isinstance(persona, dict) and 'name' in persona:
        print(f"Generating content in the style of: {persona['name']}")
    
    # Create a content brief
    content_brief = ContentBrief(
        topic=topic,
        content_type=content_type,
        target_audience="General readers",
        key_points=[
            "Introduction to the topic",
            "Main aspects or considerations",
            "Real-world applications or implications",
            "Conclusion or takeaways"
        ],
        length="500-800 words"
    )
    
    # Set style parameters (default is balanced)
    style_parameters = StyleParameters(
        style_fidelity=0.8,             # Higher means closer adherence to author's style
        vocabulary_adherence=0.7,       # Match vocabulary patterns
        sentence_structure_adherence=0.7, # Match sentence structures
        rhetorical_devices_usage=0.6,   # Use author's rhetorical devices
        tone_consistency=0.8,           # Match author's tone
        quirk_frequency=0.5,            # Include author's quirks moderately
        creative_freedom=0.3            # Mostly stick to author's patterns
    )
    
    # Initialize the agent
    agent = StyleReplicationAgent()
    
    # Generate content
    print(f"Generating {content_type} about {topic}...")
    generated_content = agent.generate_content(
        persona=persona,
        content_brief=content_brief.model_dump(),
        style_parameters=style_parameters.model_dump(),
        output_path=os.path.join(output_dir, f"{topic.replace(' ', '_')}.json"),
        format='json'
    )
    
    # Also save as Markdown for easier reading
    agent.save_generated_content(
        generated_content=generated_content,
        output_path=os.path.join(output_dir, f"{topic.replace(' ', '_')}.md"),
        format='md'
    )
    
    print(f"\nContent generated successfully!")
    print(f"Content saved to: {os.path.join(output_dir, f'{topic.replace(' ', '_')}.md')}")
    
    # Print a sample of the generated content
    content_preview = generated_content.content[:300] + "..." if len(generated_content.content) > 300 else generated_content.content
    print("\nPreview of generated content:")
    print("-" * 80)
    print(content_preview)
    print("-" * 80)
    
    return generated_content


def refine_content_example(generated_content, persona, output_dir):
    """
    Demonstrate how to refine content using feedback.
    
    Args:
        generated_content: The previously generated content
        persona: The author persona
        output_dir: Directory to save the refined output
    """
    print("\nDemonstrating content refinement with feedback...")
    
    # Create feedback
    feedback = GenerationFeedback(
        overall_rating=3,
        style_match_rating=4,
        content_quality_rating=3,
        specific_feedback=[
            "Content is good but could be more concise",
            "Need more concrete examples"
        ],
        elements_to_emphasize=[
            "Vocabulary choices",
            "Rhetorical devices"
        ],
        elements_to_reduce=[
            "Sentence complexity"
        ]
    )
    
    # Initialize the agent
    agent = StyleReplicationAgent()
    
    # Refine the content
    refined_content = agent.refine_content_with_feedback(
        persona=persona,
        content=generated_content,
        feedback=feedback.model_dump(),
        output_path=os.path.join(output_dir, f"refined_{generated_content.content_brief.topic.replace(' ', '_')}.md"),
        format='md'
    )
    
    print(f"\nContent refined successfully!")
    print(f"Refined content saved to: {os.path.join(output_dir, f'refined_{generated_content.content_brief.topic.replace(' ', '_')}.md')}")
    
    # Print a sample of the refined content
    content_preview = refined_content.content[:300] + "..." if len(refined_content.content) > 300 else refined_content.content
    print("\nPreview of refined content:")
    print("-" * 80)
    print(content_preview)
    print("-" * 80)
    
    return refined_content


def analyze_style_adherence_example(content, persona):
    """
    Demonstrate how to analyze style adherence.
    
    Args:
        content: The content to analyze
        persona: The author persona
    """
    print("\nAnalyzing style adherence...")
    
    # Initialize the agent
    agent = StyleReplicationAgent()
    
    # Prepare the persona context
    persona_context = agent._prepare_persona_context(persona)
    
    # Analyze the content
    analysis = agent._analyze_style_adherence(
        persona=persona_context,
        content=content.content
    )
    
    print("\nStyle Adherence Analysis:")
    print("-" * 80)
    
    # Print a summary of the analysis
    if "overall_score" in analysis:
        print(f"Overall Score: {analysis['overall_score']}")
    
    # Print scores for each aspect if available
    aspects = [
        "vocabulary", "sentence_structure", "rhetorical_devices", 
        "tone", "quirks"
    ]
    
    for aspect in aspects:
        for key in analysis:
            if aspect in key.lower() and "score" in key.lower():
                print(f"{key}: {analysis[key]}")
    
    print("-" * 80)
    
    return analysis


def main():
    """Run the style replication example."""
    parser = argparse.ArgumentParser(description="Style Replication Example")
    parser.add_argument("--persona", help="Path to persona JSON file", default=None)
    parser.add_argument("--topic", help="Topic for content generation", default="Artificial Intelligence Ethics")
    parser.add_argument("--content-type", help="Type of content", default="blog post")
    parser.add_argument("--output-dir", help="Output directory", default="./output")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If no persona specified, try to use a default
    persona_path = args.persona
    if persona_path is None:
        # Try to find a persona file in the output directory
        possible_files = list(Path("./output").glob("*persona.json"))
        if possible_files:
            persona_path = str(possible_files[0])
            print(f"Using found persona: {persona_path}")
        else:
            # Generate a new persona using the AgentWorkflow
            print("No persona specified. First, let's generate a persona using the AgentWorkflow.")
            
            # Ask for a directory containing sample documents
            docs_dir = input("Enter path to a directory containing sample documents: ")
            
            # Run the workflow to generate a persona
            workflow = AgentWorkflow(output_dir=str(output_dir))
            result = workflow.run(folder_path=docs_dir)
            
            if "error" in result:
                print(f"Error generating persona: {result['error']}")
                return
                
            persona_path = os.path.join(output_dir, "simulacra_persona.json")
            print(f"Persona generated: {persona_path}")
    
    # Generate content
    generated_content = generate_content(
        persona_path=persona_path,
        topic=args.topic,
        content_type=args.content_type,
        output_dir=str(output_dir)
    )
    
    # Load the persona (again to ensure correct format for subsequent steps)
    persona = load_persona(persona_path)
    
    # Demonstrate refining content with feedback
    refined_content = refine_content_example(
        generated_content=generated_content,
        persona=persona,
        output_dir=str(output_dir)
    )
    
    # Demonstrate style adherence analysis
    analyze_style_adherence_example(
        content=refined_content,
        persona=persona
    )
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
