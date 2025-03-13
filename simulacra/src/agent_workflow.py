"""
Agent Workflow Module

This module implements the workflow that coordinates agents using the handoff mechanism from the
OpenAI Agents SDK. It sets up the initial agent, configures handoffs between analysis and generation
agents, and returns the final results to the user.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Import OpenAI Agents SDK components
from .openai_agents import Agent, AgentTool, NamedAgentTool, AgentAction
# Handoff is not defined in our stub, so removing it

# Import agent components
from .document_analysis.document_analyzer import DocumentAnalysisAgent, AuthorAnalysis
from .persona_generator.persona_generation_agent import PersonaGenerationAgent, AuthorPersona
from .file_operations.directory_traversal import DirectoryTraverser
from .persona_generator.persona import Persona
from .style_replication.style_replication_agent import StyleReplicationAgent, ContentBrief, StyleParameters

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentWorkflow:
    """
    Coordinates the workflow between different agents using the handoff mechanism.
    This class sets up the initial agent, configures handoffs between analysis and generation agents,
    and returns the final results to the user.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo",
        output_dir: str = "./output",
        trace_dir: Optional[str] = None,
    ):
        """
        Initialize the agent workflow.

        Args:
            api_key: OpenAI API key. If None, it will be loaded from environment variables.
            model: The model to use for analysis and generation.
            output_dir: Directory to save output files.
            trace_dir: Directory to save trace logs. If None, tracing is disabled.
        """
        self.api_key = api_key
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up tracing if a directory is provided
        self.trace_dir = None
        if trace_dir:
            self.trace_dir = Path(trace_dir)
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Trace logs will be saved to: {self.trace_dir}")
        
        # Initialize components
        self.directory_traverser = DirectoryTraverser()
        self.document_analysis_agent = DocumentAnalysisAgent(api_key=self.api_key, model=self.model)
        self.persona_generation_agent = PersonaGenerationAgent(api_key=self.api_key, model=self.model)
        self.style_replication_agent = StyleReplicationAgent(api_key=self.api_key, model=self.model)
        
        # Set up the main entry agent
        self.entry_agent = self._create_entry_agent()
        
        logger.info("Agent workflow initialized")
    
    def _create_entry_agent(self) -> Agent:
        """
        Create the entry agent that handles user input and initiates the workflow.
        
        Returns:
            The configured entry agent
        """
        system_prompt = """You are an assistant that helps users analyze documents and generate author personas.
        You can analyze a folder of documents to understand the writing style, then generate a detailed
        persona that captures the essence of the author's voice.
        
        To get started, you need a path to a folder containing documents to analyze.
        """
        
        tools = [
            NamedAgentTool(
                name="start_document_analysis",
                description="Start the document analysis process on a folder",
                callable=self._start_document_analysis
            ),
            NamedAgentTool(
                name="validate_folder_path",
                description="Validate that a folder path exists and contains documents",
                callable=self._validate_folder_path
            ),
            NamedAgentTool(
                name="generate_content",
                description="Generate content in an author's style using their persona",
                callable=self._generate_content
            )
        ]
        
        return Agent(
            system_prompt=system_prompt,
            tools=tools,
            model=self.model,
            api_key=self.api_key
        )
    
    def _validate_folder_path(self, folder_path: str) -> Dict[str, Any]:
        """
        Validate that a folder path exists and contains documents.
        
        Args:
            folder_path: Path to the folder to validate
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating folder path: {folder_path}")
        
        folder = Path(folder_path)
        if not folder.exists():
            return {
                "valid": False,
                "error": f"Folder '{folder_path}' does not exist."
            }
        
        if not folder.is_dir():
            return {
                "valid": False,
                "error": f"'{folder_path}' is not a directory."
            }
        
        # Check for text documents
        text_extensions = ['.txt', '.md', '.rst', '.html', '.tex', '.docx']
        document_count = 0
        
        for ext in text_extensions:
            document_count += len(list(folder.glob(f"**/*{ext}")))
        
        if document_count == 0:
            return {
                "valid": False,
                "error": f"No text documents found in '{folder_path}'. Supported formats: {', '.join(text_extensions)}"
            }
        
        return {
            "valid": True,
            "document_count": document_count,
            "message": f"Found {document_count} documents in {folder_path}"
        }
    
    def _generate_content(
        self,
        persona_path: str,
        topic: str,
        content_type: str = "blog post",
        style_fidelity: float = 0.8,
        output_format: str = "md"
    ) -> Dict[str, Any]:
        """
        Generate content in the style of an author based on their persona.
        
        Args:
            persona_path: Path to a persona JSON file (either AuthorPersona or Simulacra Persona)
            topic: The topic or subject for the content
            content_type: Type of content (e.g., 'blog post', 'article', 'email')
            style_fidelity: How closely to match the author's style (0.0 to 1.0)
            output_format: Format to save output ('json', 'md', 'txt', 'html')
            
        Returns:
            Dictionary with the generated content and metadata
        """
        logger.info(f"Generating content on topic '{topic}' using persona from {persona_path}")
        
        # Check if persona file exists
        persona_file = Path(persona_path)
        if not persona_file.exists():
            return {
                "error": f"Persona file not found: {persona_path}",
                "message": "Please provide a valid path to a persona JSON file."
            }
        
        # Load the persona
        try:
            with open(persona_file, 'r', encoding='utf-8') as f:
                persona_data = json.load(f)
            
            # Try to determine if this is an AuthorPersona or a Simulacra Persona
            if 'traits' in persona_data and 'background' in persona_data:
                # This looks like a Simulacra Persona
                persona = Persona.from_dict(persona_data)
            else:
                # Use as a raw dictionary for the StyleReplicationAgent to handle
                persona = persona_data
                
        except Exception as e:
            return {
                "error": f"Failed to load persona: {str(e)}",
                "message": "The persona file could not be loaded or parsed as valid JSON."
            }
        
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
        
        # Set style parameters with the requested fidelity
        style_parameters = StyleParameters(
            style_fidelity=style_fidelity,
            vocabulary_adherence=style_fidelity * 0.9,
            sentence_structure_adherence=style_fidelity * 0.9,
            rhetorical_devices_usage=style_fidelity * 0.8,
            tone_consistency=style_fidelity * 0.9,
            quirk_frequency=style_fidelity * 0.7,
            creative_freedom=max(0.2, 1.0 - style_fidelity)
        )
        
        # Generate output filename based on topic
        filename_base = topic.replace(' ', '_').lower()
        output_path = self.output_dir / f"generated_{filename_base}"
        
        # Generate content
        try:
            logger.info(f"Generating {content_type} about {topic} with style fidelity {style_fidelity}")
            generated_content = self.style_replication_agent.generate_content(
                persona=persona,
                content_brief=content_brief.model_dump(),
                style_parameters=style_parameters.model_dump(),
                output_path=f"{output_path}.json",
                format='json'
            )
            
            # Also save in the requested format for easier reading
            content_file = self.style_replication_agent.save_generated_content(
                generated_content=generated_content,
                output_path=f"{output_path}.{output_format}",
                format=output_format
            )
            
            logger.info(f"Content generated successfully and saved to {content_file}")
            
            # Create a preview of the content
            content_preview = generated_content.content[:300] + "..." if len(generated_content.content) > 300 else generated_content.content
            
            # Return the result
            return {
                "content_preview": content_preview,
                "files": {
                    "json": str(Path(f"{output_path}.json").relative_to(self.output_dir)),
                    "readable": str(Path(f"{output_path}.{output_format}").relative_to(self.output_dir))
                },
                "metadata": {
                    "topic": topic,
                    "content_type": content_type,
                    "style_fidelity": style_fidelity,
                    "persona": persona_file.name
                },
                "message": f"Successfully generated {content_type} about '{topic}' in the author's style."
            }
            
        except Exception as e:
            logger.exception(f"Error generating content: {e}")
            return {
                "error": str(e),
                "message": "An error occurred while generating content. See logs for details."
            }
    
    def _start_document_analysis(self, folder_path: str, file_extensions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Start the document analysis process and hand off to the document analysis agent.
        
        Args:
            folder_path: Path to the folder containing documents
            file_extensions: Optional list of file extensions to include
            
        Returns:
            A handoff to the document analysis agent
        """
        logger.info(f"Starting document analysis for folder: {folder_path}")
        
        # Default extensions if none provided
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.rst', '.html', '.tex', '.docx']
        
        # Validate the folder path
        validation = self._validate_folder_path(folder_path)
        if not validation["valid"]:
            return {"error": validation["error"]}
        
        # Create a handoff to the document analysis agent
        logger.info(f"Handing off to document analysis agent for folder: {folder_path}")
        
        # Set up tracing if configured
        trace_options = None
        if self.trace_dir:
            trace_file = self.trace_dir / f"document_analysis_{Path(folder_path).stem}.jsonl"
            trace_options = {"save_path": str(trace_file)}
        
        return Handoff(
            agent=self.document_analysis_agent,
            message=f"Analyze the documents in {folder_path} with extensions {file_extensions}",
            action=AgentAction(
                tool="analyze_documents",
                tool_input={
                    "folder_path": folder_path,
                    "file_extensions": file_extensions
                }
            ),
            trace_options=trace_options
        )
    
    def _handle_document_analysis_result(self, analysis: AuthorAnalysis) -> Dict[str, Any]:
        """
        Handle the result from the document analysis agent and hand off to the persona generation agent.
        
        Args:
            analysis: The analysis results from the document analysis agent
            
        Returns:
            A handoff to the persona generation agent
        """
        logger.info("Document analysis completed, handing off to persona generation agent")
        
        # Save the analysis to a file
        analysis_path = self.output_dir / "analysis.json"
        self.document_analysis_agent.save_analysis(analysis, str(analysis_path))
        
        # Create a human-readable report
        report_path = self.output_dir / "analysis_report.md"
        self.document_analysis_agent.generate_analysis_report(analysis, str(report_path))
        
        logger.info(f"Analysis saved to {analysis_path}")
        logger.info(f"Analysis report generated at {report_path}")
        
        # Set up tracing if configured
        trace_options = None
        if self.trace_dir:
            trace_file = self.trace_dir / "persona_generation.jsonl"
            trace_options = {"save_path": str(trace_file)}
        
        # Hand off to the persona generation agent
        return Handoff(
            agent=self.persona_generation_agent,
            message="Generate a persona from the document analysis results",
            action=AgentAction(
                tool="generate_persona_from_analysis",
                tool_input={"analysis": analysis.model_dump()}
            ),
            trace_options=trace_options
        )
    
    def _handle_persona_generation_result(self, persona: AuthorPersona) -> Dict[str, Any]:
        """
        Handle the result from the persona generation agent and return the final result to the user.
        
        Args:
            persona: The persona generated by the persona generation agent
            
        Returns:
            The final result to return to the user
        """
        logger.info(f"Persona generation completed: {persona.name}")
        
        # Save the author persona to a file
        persona_path = self.output_dir / "author_persona.json"
        self.persona_generation_agent.save_persona(persona, str(persona_path))
        
        # Convert to a Simulacra Persona
        simulacra_persona = self.persona_generation_agent.convert_to_simulacra_persona(persona)
        
        # Save the Simulacra Persona to a file
        sim_persona_path = self.output_dir / "simulacra_persona.json"
        with open(sim_persona_path, 'w', encoding='utf-8') as f:
            json.dump(simulacra_persona.to_dict(), f, indent=2)
        
        logger.info(f"Author persona saved to {persona_path}")
        logger.info(f"Simulacra persona saved to {sim_persona_path}")
        
        # Generate a sample response using the persona
        sample_prompt = "What are your thoughts on technology and its impact on society?"
        response = simulacra_persona.generate_response(sample_prompt, max_tokens=150)
        
        # Return the final result
        return {
            "persona": {
                "name": persona.name,
                "writing_voice_summary": persona.writing_voice_summary[:200] + "...",
                "style_markers": persona.style_markers.structural_quirks[:3],
                "recommended_topics": persona.recommended_topics[:3]
            },
            "sample_response": response,
            "files": {
                "analysis": str(persona_path.relative_to(self.output_dir)),
                "report": str(Path("analysis_report.md").relative_to(self.output_dir)),
                "author_persona": str(persona_path.relative_to(self.output_dir)),
                "simulacra_persona": str(sim_persona_path.relative_to(self.output_dir))
            },
            "message": f"Successfully generated author persona '{persona.name}' from document analysis."
        }
    
    def run(self, folder_path: str, file_extensions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the complete workflow from document analysis to persona generation.
        
        Args:
            folder_path: Path to the folder containing documents
            file_extensions: Optional list of file extensions to include
            
        Returns:
            The final result with the generated persona and file paths
        """
        try:
            logger.info(f"Starting workflow for folder: {folder_path}")
            
            # Validate the folder path
            validation = self._validate_folder_path(folder_path)
            if not validation["valid"]:
                logger.error(f"Folder validation failed: {validation['error']}")
                return {"error": validation["error"]}
            
            # Run document analysis
            logger.info(f"Running document analysis on {validation['document_count']} documents")
            analysis = self.document_analysis_agent.analyze_documents(
                folder_path=folder_path,
                file_extensions=file_extensions or ['.txt', '.md', '.rst', '.html', '.tex', '.docx']
            )
            
            # Save analysis results
            analysis_path = self.output_dir / "analysis.json"
            self.document_analysis_agent.save_analysis(analysis, str(analysis_path))
            
            report_path = self.output_dir / "analysis_report.md"
            self.document_analysis_agent.generate_analysis_report(analysis, str(report_path))
            
            # Generate persona
            logger.info("Generating author persona from analysis")
            author_persona = self.persona_generation_agent.generate_persona_from_analysis(analysis)
            
            # Save persona
            persona_path = self.output_dir / "author_persona.json"
            self.persona_generation_agent.save_persona(author_persona, str(persona_path))
            
            # Convert to simulacra persona
            simulacra_persona = self.persona_generation_agent.convert_to_simulacra_persona(author_persona)
            
            # Save simulacra persona
            sim_persona_path = self.output_dir / "simulacra_persona.json"
            with open(sim_persona_path, 'w', encoding='utf-8') as f:
                json.dump(simulacra_persona.to_dict(), f, indent=2)
            
            # Generate a sample response
            sample_prompt = "What are your thoughts on technology and its impact on society?"
            response = simulacra_persona.generate_response(sample_prompt, max_tokens=150)
            
            # Return the final result
            return {
                "persona": {
                    "name": author_persona.name,
                    "writing_voice_summary": author_persona.writing_voice_summary[:200] + "...",
                    "style_markers": author_persona.style_markers.structural_quirks[:3],
                    "recommended_topics": author_persona.recommended_topics[:3]
                },
                "sample_response": response,
                "files": {
                    "analysis": str(analysis_path.relative_to(self.output_dir)),
                    "report": str(report_path.relative_to(self.output_dir)),
                    "author_persona": str(persona_path.relative_to(self.output_dir)),
                    "simulacra_persona": str(sim_persona_path.relative_to(self.output_dir))
                },
                "message": f"Successfully generated author persona '{author_persona.name}' from document analysis."
            }
            
        except Exception as e:
            logger.exception(f"Error in workflow: {e}")
            return {
                "error": str(e),
                "message": "An error occurred during the workflow execution. See logs for details."
            }
            
    def generate_styled_content(
        self,
        persona_path: str,
        topic: str,
        content_type: str = "blog post",
        style_fidelity: float = 0.8
    ) -> Dict[str, Any]:
        """
        Generate content in an author's style based on their persona.
        
        This is a convenience wrapper around the _generate_content method.
        
        Args:
            persona_path: Path to a persona JSON file
            topic: The topic for the content
            content_type: Type of content to generate
            style_fidelity: How closely to match the author's style (0.0 to 1.0)
            
        Returns:
            Dictionary with the generated content and metadata
        """
        return self._generate_content(
            persona_path=persona_path,
            topic=topic,
            content_type=content_type,
            style_fidelity=style_fidelity
        )
