"""
Example demonstrating the use of the PersonaGenerationAgent to generate a comprehensive
persona from document analysis results.

This example shows how to:
1. Analyze documents using the DocumentAnalysisAgent
2. Generate a persona from the analysis using the PersonaGenerationAgent
3. Convert to a simulacra Persona and use it to generate a response
4. Save the persona to disk
"""

import os
from pathlib import Path
import json

from src.document_analysis.document_analyzer import DocumentAnalysisAgent
from src.persona_generator.persona_generation_agent import PersonaGenerationAgent

# Set up API key from environment variable
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

def main():
    print("=== Persona Generation Example ===\n")
    
    # Create a DocumentAnalysisAgent
    print("Initializing Document Analysis Agent...")
    doc_agent = DocumentAnalysisAgent(api_key=api_key)
    
    # Specify the folder containing sample texts to analyze
    samples_folder = Path("./samples")
    if not samples_folder.exists():
        # Create samples folder if it doesn't exist
        samples_folder.mkdir()
        print(f"Created samples directory at {samples_folder.absolute()}")
        print("Please add sample text files before running this example.")
        return
    
    # Check if there are any text files in the samples folder
    text_files = list(samples_folder.glob("*.txt")) + list(samples_folder.glob("*.md"))
    if not text_files:
        print(f"No text files found in {samples_folder}. Please add some .txt or .md files.")
        return
    
    print(f"Found {len(text_files)} text file(s) to analyze.")
    
    # Analyze the documents
    print("\nAnalyzing documents...")
    analysis = doc_agent.analyze_documents(str(samples_folder))
    
    # Print a summary of the analysis
    print("\nAnalysis complete! Summary:")
    print(f"- Document count: {analysis.features.document_count}")
    print(f"- Total words analyzed: {analysis.features.total_word_count}")
    print(f"- Vocabulary size: {analysis.features.vocabulary_size}")
    
    # Extract a brief snippet of the writing style summary
    summary_snippet = analysis.writing_style_summary.split('.')[0] + "..."
    print(f"- Writing style: {summary_snippet}")
    
    # Emotional profile snippet
    print(f"- Emotional expressiveness: {analysis.psychological_profile.emotional_expressiveness:.2f}")
    print(f"- Analytical thinking: {analysis.psychological_profile.analytical_thinking:.2f}")
    
    # Save the analysis to a file
    analysis_path = Path("./output/analysis.json")
    analysis_path.parent.mkdir(exist_ok=True)
    doc_agent.save_analysis(analysis, str(analysis_path))
    print(f"\nSaved analysis to {analysis_path}")
    
    # Create a human-readable report
    report_path = Path("./output/analysis_report.md")
    doc_agent.generate_analysis_report(analysis, str(report_path))
    print(f"Generated human-readable report at {report_path}")
    
    # Create a PersonaGenerationAgent
    print("\nInitializing Persona Generation Agent...")
    persona_agent = PersonaGenerationAgent(api_key=api_key)
    
    # Generate a persona from the analysis
    print("Generating author persona from analysis...")
    author_persona = persona_agent.generate_persona_from_analysis(analysis)
    
    # Print the persona details
    print("\nGenerated Author Persona:")
    print(f"- Name: {author_persona.name}")
    print(f"- Writing Voice: {author_persona.writing_voice_summary[:100]}...")
    
    # Print some style markers
    print("\nStyle Markers:")
    for i, quirk in enumerate(author_persona.style_markers.structural_quirks[:3]):
        print(f"- {quirk}")
    
    # Print recommended topics
    print("\nRecommended Topics:")
    for i, topic in enumerate(author_persona.recommended_topics[:3]):
        print(f"- {topic}")
    
    # Save the author persona to a file
    persona_path = Path("./output/author_persona.json")
    persona_agent.save_persona(author_persona, str(persona_path))
    print(f"\nSaved author persona to {persona_path}")
    
    # Convert to a Simulacra Persona
    print("\nConverting to Simulacra Persona...")
    simulacra_persona = persona_agent.convert_to_simulacra_persona(author_persona)
    
    # Print the Simulacra Persona details
    print("\nSimulacra Persona:")
    print(f"- Name: {simulacra_persona.name}")
    print(f"- Traits: {', '.join(simulacra_persona.traits[:3])}...")
    print(f"- Communication style: {simulacra_persona.communication_style}")
    print(f"- Knowledge areas: {', '.join(simulacra_persona.knowledge_areas[:3])}...")
    
    # Save the Simulacra Persona to a file
    sim_persona_path = Path("./output/simulacra_persona.json")
    with open(sim_persona_path, 'w', encoding='utf-8') as f:
        json.dump(simulacra_persona.to_dict(), f, indent=2)
    print(f"\nSaved Simulacra persona to {sim_persona_path}")
    
    # Generate a sample response using the persona
    print("\nGenerating a sample response using the persona...")
    sample_prompt = "What are your thoughts on technology and its impact on society?"
    response = simulacra_persona.generate_response(sample_prompt, max_tokens=150)
    
    print("\nSample response from the persona:")
    print(f"\"{response}\"")
    
    print("\n=== Example Complete ===")

if __name__ == "__main__":
    main()
