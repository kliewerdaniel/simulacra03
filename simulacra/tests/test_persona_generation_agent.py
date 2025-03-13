import os
import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.document_analysis.document_analyzer import AuthorAnalysis, DocumentFeatures, PsychologicalProfile
from src.persona_generator.persona_generation_agent import (
    PersonaGenerationAgent, 
    WritingCharacteristics, 
    StyleMarkers, 
    PsychologicalTraits,
    AuthorPersona
)
from src.persona_generator.persona import Persona


# Mock data for testing
@pytest.fixture
def mock_author_analysis():
    # Create a minimal valid AuthorAnalysis object
    features = DocumentFeatures(
        vocabulary_size=5000,
        average_word_length=4.7,
        word_frequencies={"the": 1000, "and": 500},
        rare_words=["esoteric", "ephemeral"],
        average_sentence_length=15.5,
        sentence_length_variation=5.2,
        sentence_structures={"simple": 30, "compound": 20, "complex": 10},
        idioms=["under the weather", "break a leg"],
        metaphors=["time is money", "life is a journey"],
        transition_phrases=["however", "in addition", "consequently"],
        paragraph_structure={"topic_sentence_position": "beginning", "avg_sentences": 5},
        punctuation_usage={",": 200, ".": 150, ";": 10},
        passive_voice_frequency=0.15,
        active_voice_frequency=0.85,
        document_count=2,
        total_word_count=10000,
        total_sentence_count=650
    )
    
    psychological_profile = PsychologicalProfile(
        openness=0.75,
        conscientiousness=0.68,
        extraversion=0.45,
        agreeableness=0.82,
        neuroticism=0.30,
        formality_level=0.65,
        analytical_thinking=0.72,
        emotional_expressiveness=0.38,
        confidence_level=0.70,
        dominant_cognitive_patterns=["analytical", "systematic", "detail-oriented"],
        communication_preferences=["direct", "precise", "structured"],
        thinking_style="logical"
    )
    
    return AuthorAnalysis(
        features=features,
        psychological_profile=psychological_profile,
        writing_style_summary="The author demonstrates a formal, analytical writing style with careful attention to detail. Sentences are moderate in length with varied structure, creating a rhythmic flow. Vocabulary is diverse, with precise word choices that reveal a technical background.",
        distinguishing_characteristics=[
            "Strong preference for active voice",
            "Frequent use of semicolons and parenthetical asides",
            "Technical vocabulary with occasional rare words",
            "Logical paragraph structures with clear topic sentences"
        ],
        recommendations=[
            "Incorporate more varied sentence openings to enhance flow",
            "Consider adding more metaphors to illustrate complex points",
            "Experiment with shorter paragraphs for increased readability"
        ]
    )


# Mock API responses for the OpenAI client calls
@pytest.fixture
def mock_openai_response():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "test_key": "test_value"
    })
    return mock_response


@pytest.fixture
def mock_name_response():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Dr. Alexandra Mercer"
    return mock_response


@pytest.fixture
def mock_summary_response():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This author writes with precision and clarity, favoring logical structures and technical terminology. Their analytical approach is balanced with occasional metaphorical expressions, creating a distinctive voice that is both authoritative and accessible."
    return mock_response


class TestPersonaGenerationAgent:
    
    def test_init(self):
        """Test that the agent initializes properly"""
        agent = PersonaGenerationAgent(api_key="test_key")
        assert agent.api_key == "test_key"
        assert agent.model == "gpt-4-turbo"
        assert agent.agent is not None
    
    @patch("openai.OpenAI")
    def test_extract_writing_characteristics(self, mock_openai, mock_openai_response):
        # Set up mock OpenAI client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.return_value = mock_client
        
        # Create agent and test
        agent = PersonaGenerationAgent(api_key="test_key")
        result = agent._extract_writing_characteristics("{}")
        
        assert result == {"test_key": "test_value"}
        mock_client.chat.completions.create.assert_called_once()
    
    @patch("openai.OpenAI")
    def test_identify_style_markers(self, mock_openai, mock_openai_response):
        # Set up mock OpenAI client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.return_value = mock_client
        
        # Create agent and test
        agent = PersonaGenerationAgent(api_key="test_key")
        result = agent._identify_style_markers("{}")
        
        assert result == {"test_key": "test_value"}
        mock_client.chat.completions.create.assert_called_once()
    
    @patch("openai.OpenAI")
    def test_analyze_psychological_traits(self, mock_openai, mock_openai_response):
        # Set up mock OpenAI client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.return_value = mock_client
        
        # Create agent and test
        agent = PersonaGenerationAgent(api_key="test_key")
        result = agent._analyze_psychological_traits("{}")
        
        assert result == {"test_key": "test_value"}
        mock_client.chat.completions.create.assert_called_once()
    
    @patch("openai.OpenAI")
    def test_generate_author_background(self, mock_openai, mock_openai_response):
        # Set up mock OpenAI client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.return_value = mock_client
        
        # Create agent and test
        agent = PersonaGenerationAgent(api_key="test_key")
        result = agent._generate_author_background("{}")
        
        assert result == {"test_key": "test_value"}
        mock_client.chat.completions.create.assert_called_once()
    
    @patch("openai.OpenAI")
    def test_suggest_topics(self, mock_openai, mock_openai_response):
        # Set up mock OpenAI client with a list response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([
            {"topic": "Artificial Intelligence", "rationale": "test rationale"},
            {"topic": "Data Science", "rationale": "test rationale"}
        ])
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create agent and test
        agent = PersonaGenerationAgent(api_key="test_key")
        result = agent._suggest_topics("{}")
        
        assert result == ["Artificial Intelligence", "Data Science"]
        mock_client.chat.completions.create.assert_called_once()
    
    @patch("openai.OpenAI")
    def test_generate_name(self, mock_openai, mock_name_response):
        # Set up mock OpenAI client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_name_response
        mock_openai.return_value = mock_client
        
        # Create agent and test
        agent = PersonaGenerationAgent(api_key="test_key")
        result = agent._generate_name({})
        
        assert result == "Dr. Alexandra Mercer"
        mock_client.chat.completions.create.assert_called_once()
    
    @patch("openai.OpenAI")
    def test_generate_writing_voice_summary(self, mock_openai, mock_summary_response):
        # Set up mock OpenAI client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_summary_response
        mock_openai.return_value = mock_client
        
        # Create agent and test
        agent = PersonaGenerationAgent(api_key="test_key")
        result = agent._generate_writing_voice_summary({})
        
        assert "analytical approach" in result
        mock_client.chat.completions.create.assert_called_once()
    
    @patch.object(PersonaGenerationAgent, "_extract_writing_characteristics")
    @patch.object(PersonaGenerationAgent, "_identify_style_markers")
    @patch.object(PersonaGenerationAgent, "_analyze_psychological_traits")
    @patch.object(PersonaGenerationAgent, "_generate_author_background")
    @patch.object(PersonaGenerationAgent, "_suggest_topics")
    @patch.object(PersonaGenerationAgent, "_generate_name")
    @patch.object(PersonaGenerationAgent, "_generate_writing_voice_summary")
    def test_generate_persona_from_analysis(
        self, 
        mock_generate_writing_voice_summary,
        mock_generate_name,
        mock_suggest_topics,
        mock_generate_author_background,
        mock_analyze_psychological_traits,
        mock_identify_style_markers,
        mock_extract_writing_characteristics,
        mock_author_analysis
    ):
        # Set up mocks
        mock_extract_writing_characteristics.return_value = {"vocabulary_profile": {}, "sentence_construction": {}}
        mock_identify_style_markers.return_value = {
            "signature_phrases": [], 
            "punctuation_patterns": {}, 
            "transition_preferences": [],
            "structural_quirks": ["Uses semicolons extensively"],
            "lexical_preferences": {}
        }
        mock_analyze_psychological_traits.return_value = {
            "personality_dimensions": {"openness": 0.8},
            "cognitive_style": {},
            "emotional_patterns": {},
            "values_indicators": [],
            "social_orientation": {}
        }
        mock_generate_author_background.return_value = {"education": "Ph.D. in Computer Science"}
        mock_suggest_topics.return_value = ["AI Ethics", "Data Privacy"]
        mock_generate_name.return_value = "Dr. Alexandra Mercer"
        mock_generate_writing_voice_summary.return_value = "Analytical and precise writing style."
        
        # Create agent and test
        agent = PersonaGenerationAgent(api_key="test_key")
        result = agent.generate_persona_from_analysis(mock_author_analysis)
        
        # Verify result
        assert isinstance(result, AuthorPersona)
        assert result.name == "Dr. Alexandra Mercer"
        assert result.writing_voice_summary == "Analytical and precise writing style."
        assert "Uses semicolons extensively" in result.style_markers.structural_quirks
        assert result.psychological_traits.personality_dimensions["openness"] == 0.8
        assert result.author_background["education"] == "Ph.D. in Computer Science"
        assert "AI Ethics" in result.recommended_topics
    
    def test_save_persona(self, tmp_path):
        """Test that the agent can save a persona to disk"""
        # Create a minimal AuthorPersona
        writing_characteristics = WritingCharacteristics(
            vocabulary_profile={}, 
            sentence_construction={}
        )
        style_markers = StyleMarkers(
            signature_phrases=[],
            punctuation_patterns={},
            transition_preferences=[],
            structural_quirks=[],
            lexical_preferences={}
        )
        psychological_traits = PsychologicalTraits(
            personality_dimensions={},
            cognitive_style={},
            emotional_patterns={},
            values_indicators=[],
            social_orientation={}
        )
        
        persona = AuthorPersona(
            name="Test Author",
            writing_characteristics=writing_characteristics,
            style_markers=style_markers,
            psychological_traits=psychological_traits,
            writing_voice_summary="Test summary",
            recommended_topics=["Topic 1", "Topic 2"],
            author_background={"education": "Test education"}
        )
        
        # Save the persona
        output_path = tmp_path / "test_persona.json"
        agent = PersonaGenerationAgent(api_key="test_key")
        saved_path = agent.save_persona(persona, str(output_path))
        
        # Verify the file was saved
        assert os.path.exists(saved_path)
        
        # Verify the contents
        with open(saved_path, 'r') as f:
            saved_data = json.load(f)
            
        assert saved_data["name"] == "Test Author"
        assert saved_data["writing_voice_summary"] == "Test summary"
        assert "Topic 1" in saved_data["recommended_topics"]
        assert saved_data["author_background"]["education"] == "Test education"
    
    @patch.object(PersonaGenerationAgent, "_extract_writing_characteristics")
    @patch.object(PersonaGenerationAgent, "_identify_style_markers")
    @patch.object(PersonaGenerationAgent, "_analyze_psychological_traits")
    @patch.object(PersonaGenerationAgent, "_generate_author_background")
    @patch.object(PersonaGenerationAgent, "_suggest_topics")
    @patch.object(PersonaGenerationAgent, "_generate_name")
    @patch.object(PersonaGenerationAgent, "_generate_writing_voice_summary")
    def test_convert_to_simulacra_persona(
        self, 
        mock_generate_writing_voice_summary,
        mock_generate_name,
        mock_suggest_topics,
        mock_generate_author_background,
        mock_analyze_psychological_traits,
        mock_identify_style_markers,
        mock_extract_writing_characteristics,
        mock_author_analysis
    ):
        # Set up mocks
        mock_extract_writing_characteristics.return_value = {"vocabulary_profile": {"richness": "high"}, "sentence_construction": {}}
        mock_identify_style_markers.return_value = {
            "signature_phrases": ["In conclusion"], 
            "punctuation_patterns": {"semicolon": "frequent"}, 
            "transition_preferences": ["however", "moreover"],
            "structural_quirks": ["Uses semicolons extensively"],
            "lexical_preferences": {"technical_terms": "high"}
        }
        mock_analyze_psychological_traits.return_value = {
            "personality_dimensions": {"openness": 0.8, "conscientiousness": 0.2},
            "cognitive_style": {"analytical": "high"},
            "emotional_patterns": {"expressiveness": "low"},
            "values_indicators": ["accuracy", "precision"],
            "social_orientation": {"communication_preferences": ["direct", "formal", "structured"]}
        }
        mock_generate_author_background.return_value = {"education": "Ph.D. in Computer Science"}
        mock_suggest_topics.return_value = ["AI Ethics", "Data Privacy", "Machine Learning Theory"]
        mock_generate_name.return_value = "Dr. Alexandra Mercer"
        mock_generate_writing_voice_summary.return_value = "Analytical and precise writing style."
        
        # Create agent and test
        agent = PersonaGenerationAgent(api_key="test_key")
        author_persona = agent.generate_persona_from_analysis(mock_author_analysis)
        result = agent.convert_to_simulacra_persona(author_persona)
        
        # Verify result
        assert isinstance(result, Persona)
        assert result.name == "Dr. Alexandra Mercer"
        assert "high openness" in result.traits
        assert "low conscientiousness" in result.traits
        assert "Uses semicolons extensively" in result.traits
        assert result.communication_style == "direct, formal, structured"
        assert "AI Ethics" in result.knowledge_areas
        assert "Data Privacy" in result.knowledge_areas
        assert "Machine Learning Theory" in result.knowledge_areas
        assert result.additional_details["writing_voice_summary"] == "Analytical and precise writing style."
        assert result.additional_details["vocabulary_profile"]["richness"] == "high"
