"""
Tests for the Agent Workflow module.

This module tests the workflow that coordinates agents using the handoff mechanism
from the Agents SDK, focusing on the handoffs between document analysis agent and
persona generation agent.
"""

import os
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agent_workflow import AgentWorkflow
from src.document_analysis.document_analyzer import AuthorAnalysis, DocumentFeatures, PsychologicalProfile
from src.persona_generator.persona_generation_agent import AuthorPersona, WritingCharacteristics, StyleMarkers, PsychologicalTraits
from src.persona_generator.persona import Persona


class TestAgentWorkflow(unittest.TestCase):
    """Test cases for the AgentWorkflow class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create temporary output directory
        self.output_dir = Path("./test_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create temporary trace directory
        self.trace_dir = Path("./test_traces")
        self.trace_dir.mkdir(exist_ok=True)
        
        # Set up the workflow with mocked agents
        self.workflow = AgentWorkflow(
            api_key="test_api_key",
            model="gpt-4-turbo",
            output_dir=str(self.output_dir),
            trace_dir=str(self.trace_dir)
        )
        
        # Create a sample folder for testing
        self.test_folder = Path("./test_documents")
        self.test_folder.mkdir(exist_ok=True)
        
        # Create a sample document
        sample_doc = self.test_folder / "sample.txt"
        with open(sample_doc, "w") as f:
            f.write("This is a sample document for testing the agent workflow.")

    def tearDown(self):
        """Clean up test environment after each test method."""
        # Remove test files and directories
        for file in self.output_dir.glob("*"):
            file.unlink()
        self.output_dir.rmdir()
        
        for file in self.trace_dir.glob("*"):
            file.unlink()
        self.trace_dir.rmdir()
        
        for file in self.test_folder.glob("*"):
            file.unlink()
        self.test_folder.rmdir()

    def test_validate_folder_path(self):
        """Test folder path validation."""
        # Test with valid folder
        result = self.workflow._validate_folder_path(str(self.test_folder))
        self.assertTrue(result["valid"])
        self.assertIn("document_count", result)
        
        # Test with non-existent folder
        result = self.workflow._validate_folder_path("./non_existent_folder")
        self.assertFalse(result["valid"])
        self.assertIn("error", result)
        
        # Test with a file instead of folder
        sample_file = Path("./sample_file.txt")
        with open(sample_file, "w") as f:
            f.write("This is not a folder")
        
        result = self.workflow._validate_folder_path(str(sample_file))
        self.assertFalse(result["valid"])
        
        # Clean up
        sample_file.unlink()

    @patch("src.document_analysis.document_analyzer.DocumentAnalysisAgent.analyze_documents")
    def test_document_analysis_handoff(self, mock_analyze_documents):
        """Test handoff to document analysis agent."""
        # Mock the analyze_documents method
        mock_features = DocumentFeatures(
            vocabulary_size=1000,
            average_word_length=4.5,
            word_frequencies={"test": 10, "sample": 5},
            rare_words=["unique", "rare"],
            average_sentence_length=10.5,
            sentence_length_variation=2.0,
            sentence_structures={"simple": 5, "complex": 3},
            idioms=["out of the blue"],
            metaphors=["time flies"],
            transition_phrases=["in addition", "therefore"],
            paragraph_structure={"avg_length": 3},
            punctuation_usage={",": 15, ".": 10},
            passive_voice_frequency=0.2,
            active_voice_frequency=0.8,
            document_count=1,
            total_word_count=500,
            total_sentence_count=50
        )
        
        mock_profile = PsychologicalProfile(
            openness=0.7,
            conscientiousness=0.8,
            extraversion=0.4,
            agreeableness=0.6,
            neuroticism=0.3,
            formality_level=0.65,
            analytical_thinking=0.85,
            emotional_expressiveness=0.4,
            confidence_level=0.75,
            dominant_cognitive_patterns=["analytical", "systematic"],
            communication_preferences=["direct", "factual"],
            thinking_style="logical"
        )
        
        mock_analysis = AuthorAnalysis(
            features=mock_features,
            psychological_profile=mock_profile,
            writing_style_summary="This author has a formal, analytical writing style.",
            distinguishing_characteristics=["Prefers active voice", "Uses complex vocabulary"],
            recommendations=["Vary sentence structure more", "Add emotional elements"]
        )
        
        mock_analyze_documents.return_value = mock_analysis
        
        # Test start_document_analysis method which creates a handoff
        handoff = self.workflow._start_document_analysis(str(self.test_folder))
        
        # Verify handoff is created correctly
        self.assertIn("agent", handoff)
        self.assertIn("message", handoff)
        self.assertIn("action", handoff)
        self.assertEqual(handoff["action"]["tool"], "analyze_documents")
        self.assertEqual(handoff["action"]["tool_input"]["folder_path"], str(self.test_folder))
        
        # Test handling of analysis results
        persona_handoff = self.workflow._handle_document_analysis_result(mock_analysis)
        
        # Verify persona generation handoff
        self.assertIn("agent", persona_handoff)
        self.assertIn("message", persona_handoff)
        self.assertIn("action", persona_handoff)
        self.assertEqual(persona_handoff["action"]["tool"], "generate_persona_from_analysis")
        
        # Check that analysis files were created
        self.assertTrue((self.output_dir / "analysis.json").exists())
        self.assertTrue((self.output_dir / "analysis_report.md").exists())

    @patch("src.persona_generator.persona_generation_agent.PersonaGenerationAgent.generate_persona_from_analysis")
    @patch("src.persona_generator.persona_generation_agent.PersonaGenerationAgent.convert_to_simulacra_persona")
    def test_persona_generation_handoff(self, mock_convert_to_simulacra, mock_generate_persona):
        """Test handoff to persona generation agent and final result handling."""
        # Mock the generate_persona method
        mock_writing_characteristics = WritingCharacteristics(
            vocabulary_profile={"complexity": "high", "richness": "moderate"},
            sentence_construction={"length": "moderate", "complexity": "high"},
            rhetorical_devices=["metaphor", "simile"],
            tone_patterns={"formal": 0.7, "casual": 0.3},
            organizational_patterns=["top-down", "thesis-first"]
        )
        
        mock_style_markers = StyleMarkers(
            signature_phrases=["in essence", "fundamentally"],
            punctuation_patterns={"em_dash": "frequent", "semicolon": "occasional"},
            transition_preferences=["moreover", "consequently"],
            structural_quirks=["starts paragraphs with questions", "ends with single-sentence summary"],
            lexical_preferences={"academic": "high", "technical": "moderate"}
        )
        
        mock_psychological_traits = PsychologicalTraits(
            personality_dimensions={"openness": 0.8, "conscientiousness": 0.9},
            cognitive_style={"analytical": "high", "intuitive": "low"},
            emotional_patterns={"expressiveness": 0.3, "positivity": 0.6},
            values_indicators=["knowledge", "precision", "clarity"],
            social_orientation={"formality": "high", "directness": "high"}
        )
        
        mock_persona = AuthorPersona(
            name="Dr. Alex Morgan",
            writing_characteristics=mock_writing_characteristics,
            style_markers=mock_style_markers,
            psychological_traits=mock_psychological_traits,
            writing_voice_summary="An academic voice marked by precision and clarity, with formal tone and complex structures.",
            recommended_topics=["scientific methodology", "cognitive science", "technical writing"],
            author_background={"education": "PhD", "profession": "academic"}
        )
        
        mock_generate_persona.return_value = mock_persona
        
        # Mock the simulacra persona
        mock_simulacra_persona = Persona(
            name="Dr. Alex Morgan",
            traits=["analytical", "precise", "formal"],
            background="PhD in academic field with focus on technical writing",
            communication_style="Formal and structured, with emphasis on clarity",
            knowledge_areas=["scientific methodology", "cognitive science", "technical writing"]
        )
        
        # Mock generate_response method
        mock_simulacra_persona.generate_response = MagicMock(return_value="Technology will likely have a profound impact on society...")
        
        mock_convert_to_simulacra.return_value = mock_simulacra_persona
        
        # Create a mock analysis for input
        mock_features = DocumentFeatures(
            vocabulary_size=1000,
            average_word_length=4.5,
            word_frequencies={"test": 10, "sample": 5},
            rare_words=["unique", "rare"],
            average_sentence_length=10.5,
            sentence_length_variation=2.0,
            sentence_structures={"simple": 5, "complex": 3},
            idioms=["out of the blue"],
            metaphors=["time flies"],
            transition_phrases=["in addition", "therefore"],
            paragraph_structure={"avg_length": 3},
            punctuation_usage={",": 15, ".": 10},
            passive_voice_frequency=0.2,
            active_voice_frequency=0.8,
            document_count=1,
            total_word_count=500,
            total_sentence_count=50
        )
        
        mock_profile = PsychologicalProfile(
            openness=0.7,
            conscientiousness=0.8,
            extraversion=0.4,
            agreeableness=0.6,
            neuroticism=0.3,
            formality_level=0.65,
            analytical_thinking=0.85,
            emotional_expressiveness=0.4,
            confidence_level=0.75,
            dominant_cognitive_patterns=["analytical", "systematic"],
            communication_preferences=["direct", "factual"],
            thinking_style="logical"
        )
        
        mock_analysis = AuthorAnalysis(
            features=mock_features,
            psychological_profile=mock_profile,
            writing_style_summary="This author has a formal, analytical writing style.",
            distinguishing_characteristics=["Prefers active voice", "Uses complex vocabulary"],
            recommendations=["Vary sentence structure more", "Add emotional elements"]
        )
        
        # Test handling of persona generation results
        result = self.workflow._handle_persona_generation_result(mock_persona)
        
        # Verify the final result structure
        self.assertIn("persona", result)
        self.assertIn("sample_response", result)
        self.assertIn("files", result)
        self.assertIn("message", result)
        
        # Check that persona files were created
        self.assertTrue((self.output_dir / "author_persona.json").exists())
        self.assertTrue((self.output_dir / "simulacra_persona.json").exists())
        
        # Test the full workflow with mocks
        with patch.object(self.workflow, '_validate_folder_path') as mock_validate:
            mock_validate.return_value = {"valid": True, "document_count": 1}
            
            with patch.object(self.workflow.document_analysis_agent, 'analyze_documents') as mock_analyze:
                mock_analyze.return_value = mock_analysis
                
                with patch.object(self.workflow.persona_generation_agent, 'generate_persona_from_analysis') as mock_gen:
                    mock_gen.return_value = mock_persona
                    
                    with patch.object(self.workflow.persona_generation_agent, 'convert_to_simulacra_persona') as mock_convert:
                        mock_convert.return_value = mock_simulacra_persona
                        
                        # Run the workflow
                        result = self.workflow.run(str(self.test_folder))
                        
                        # Verify the result
                        self.assertIn("persona", result)
                        self.assertIn("sample_response", result)
                        self.assertIn("files", result)
                        self.assertEqual(result["persona"]["name"], "Dr. Alex Morgan")


if __name__ == '__main__':
    unittest.main()
