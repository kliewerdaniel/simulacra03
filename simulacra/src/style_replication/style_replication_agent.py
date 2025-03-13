"""
Style Replication Agent Module

This module implements an agent that generates content based on an author's style persona.
The agent can control how closely the output matches the original style and incorporates
feedback to refine the generated content.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

from ..openai_agents import Agent, AgentTool, NamedAgentTool, AgentAction
from pydantic import BaseModel, Field, validator

from ..persona_generator.persona import Persona
from ..persona_generator.persona_generation_agent import AuthorPersona

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContentBrief(BaseModel):
    """A brief describing the content to be generated."""
    
    topic: str = Field(description="The main topic or subject of the content")
    
    content_type: str = Field(
        description="The type of content to generate (e.g., blog post, article, email, social media post)"
    )
    
    target_audience: str = Field(
        description="The intended audience for the content",
        default="General"
    )
    
    key_points: List[str] = Field(
        description="Key points to cover in the content",
        default_factory=list
    )
    
    tone: Optional[str] = Field(
        description="Desired tone for the content (if different from author's natural tone)",
        default=None
    )
    
    length: str = Field(
        description="Approximate length of the content (e.g., 'short', '500 words', '2 paragraphs')",
        default="medium"
    )
    
    additional_requirements: Dict[str, Any] = Field(
        description="Any additional requirements or constraints",
        default_factory=dict
    )


class StyleParameters(BaseModel):
    """Parameters controlling how closely the output matches the original style."""
    
    style_fidelity: float = Field(
        description="How closely to match the author's style (0.0 to 1.0)",
        default=0.8
    )
    
    vocabulary_adherence: float = Field(
        description="How closely to match the author's vocabulary patterns (0.0 to 1.0)",
        default=0.7
    )
    
    sentence_structure_adherence: float = Field(
        description="How closely to match the author's sentence structures (0.0 to 1.0)",
        default=0.7
    )
    
    rhetorical_devices_usage: float = Field(
        description="How frequently to use the author's rhetorical devices (0.0 to 1.0)",
        default=0.6
    )
    
    tone_consistency: float = Field(
        description="How closely to match the author's typical tone (0.0 to 1.0)",
        default=0.8
    )
    
    quirk_frequency: float = Field(
        description="How often to incorporate the author's quirks/idiosyncrasies (0.0 to 1.0)",
        default=0.5
    )
    
    creative_freedom: float = Field(
        description="How much creative freedom to allow beyond the author's established patterns (0.0 to 1.0)",
        default=0.3
    )
    
    @validator('style_fidelity', 'vocabulary_adherence', 'sentence_structure_adherence', 
               'rhetorical_devices_usage', 'tone_consistency', 'quirk_frequency', 'creative_freedom')
    def validate_range(cls, v, values, **kwargs):
        """Validate that all parameters are between 0 and 1."""
        if not 0 <= v <= 1:
            field_name = kwargs['field'].name
            raise ValueError(f"{field_name} must be between 0.0 and 1.0")
        return v


class GenerationFeedback(BaseModel):
    """Feedback on generated content to guide refinement."""
    
    overall_rating: int = Field(
        description="Overall rating of the generated content (1-5)",
        ge=1,
        le=5
    )
    
    style_match_rating: int = Field(
        description="How well the content matches the author's style (1-5)",
        ge=1,
        le=5
    )
    
    content_quality_rating: int = Field(
        description="Quality of the content regardless of style (1-5)",
        ge=1,
        le=5
    )
    
    specific_feedback: List[str] = Field(
        description="Specific feedback points for refinement",
        default_factory=list
    )
    
    elements_to_emphasize: List[str] = Field(
        description="Stylistic elements to emphasize more",
        default_factory=list
    )
    
    elements_to_reduce: List[str] = Field(
        description="Stylistic elements to tone down",
        default_factory=list
    )
    
    @validator('overall_rating', 'style_match_rating', 'content_quality_rating')
    def validate_rating(cls, v):
        """Validate that ratings are between 1 and 5."""
        if not 1 <= v <= 5:
            raise ValueError(f"Rating must be between 1 and 5")
        return v


class GeneratedContent(BaseModel):
    """Generated content with metadata and refinement history."""
    
    content: str = Field(description="The generated content text")
    
    content_brief: ContentBrief = Field(
        description="The brief used to generate the content"
    )
    
    style_parameters: StyleParameters = Field(
        description="The style parameters used for generation"
    )
    
    refinement_history: List[Dict[str, Any]] = Field(
        description="History of refinements and feedback",
        default_factory=list
    )
    
    metadata: Dict[str, Any] = Field(
        description="Additional metadata about the generation process",
        default_factory=dict
    )


class StyleReplicationAgent:
    """
    An agent that generates content in the style of an author based on their persona.
    
    This agent takes a persona (either AuthorPersona or Simulacra Persona) and a content
    brief, then generates text that matches the author's style. It also supports:
    
    1. Style parameter controls to adjust how closely to match different aspects of the style
    2. Feedback mechanisms to refine the generated content
    3. Versioning and tracking of refinements
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo",
    ):
        """
        Initialize the style replication agent.
        
        Args:
            api_key: OpenAI API key. If None, it will be loaded from environment variables.
            model: The model to use for content generation.
        """
        self.api_key = api_key
        self.model = model
        
        # Initialize the agent with tools
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """Create an OpenAI Agent with tools for style replication."""
        
        tools = [
            NamedAgentTool(
                name="generate_content_draft",
                description="Generate a draft of content in the author's style",
                callable=self._generate_content_draft,
            ),
            NamedAgentTool(
                name="refine_content",
                description="Refine content based on feedback",
                callable=self._refine_content,
            ),
            NamedAgentTool(
                name="analyze_style_adherence",
                description="Analyze how well content adheres to the author's style",
                callable=self._analyze_style_adherence,
            ),
        ]
        
        system_prompt = """You are an expert style replication agent specializing in generating content that 
        matches an author's unique writing style. Your task is to take an author persona and a content brief, 
        then generate text that authentically captures the author's voice, stylistic patterns, and quirks.
        
        When generating content:
        1. Pay careful attention to vocabulary choices, sentence structures, and rhetorical patterns
        2. Incorporate the author's distinctive stylistic markers
        3. Match the author's typical tone while adapting to the content requirements
        4. Balance staying true to the author's style with meeting the content brief's needs
        
        You have tools available to:
        1. Generate initial content drafts in the author's style
        2. Refine content based on feedback
        3. Analyze how well content adheres to the author's style
        
        Create content that convincingly appears to have been written by the original author while 
        addressing the specific requirements in the content brief.
        """
        
        return Agent(
            system_prompt=system_prompt,
            tools=tools,
            model=self.model,
            api_key=self.api_key
        )
    
    def _prepare_persona_context(self, persona: Any) -> Dict[str, Any]:
        """
        Convert either an AuthorPersona or Persona object to a standardized format for the agent.
        
        Args:
            persona: Either an AuthorPersona or Simulacra Persona object
            
        Returns:
            A dictionary with standardized persona information
        """
        persona_context = {}
        
        # Handle AuthorPersona
        if isinstance(persona, AuthorPersona):
            persona_context = {
                "name": persona.name,
                "writing_characteristics": persona.writing_characteristics.model_dump(),
                "style_markers": persona.style_markers.model_dump(),
                "psychological_traits": persona.psychological_traits.model_dump(),
                "writing_voice_summary": persona.writing_voice_summary,
                "recommended_topics": persona.recommended_topics,
                "author_background": persona.author_background
            }
        # Handle Simulacra Persona
        elif isinstance(persona, Persona):
            persona_context = {
                "name": persona.name,
                "traits": persona.traits,
                "background": persona.background,
                "communication_style": persona.communication_style,
                "knowledge_areas": persona.knowledge_areas,
                "additional_details": persona.additional_details,
                "system_message": persona._generate_system_message()
            }
        # Handle dictionary format (already serialized)
        elif isinstance(persona, dict):
            persona_context = persona
        else:
            raise ValueError(f"Unsupported persona type: {type(persona)}")
            
        return persona_context
    
    def _generate_content_draft(
        self,
        persona: Dict[str, Any],
        content_brief: Dict[str, Any],
        style_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a draft of content in the author's style.
        
        Args:
            persona: The author persona in dictionary format
            content_brief: The content brief in dictionary format
            style_parameters: Optional style parameters to control generation
            
        Returns:
            A dictionary with the generated content
        """
        import requests
        
        # Convert dictionaries to models if needed
        if isinstance(content_brief, dict):
            content_brief = ContentBrief(**content_brief)
            
        if style_parameters is None:
            style_parameters = StyleParameters().model_dump()
        elif isinstance(style_parameters, dict):
            style_parameters = StyleParameters(**style_parameters)
            
        # Prepare the system prompt based on the persona and style parameters
        system_prompt = self._create_style_replication_prompt(persona, style_parameters)
        
        # Prepare the user content request
        content_request = self._create_content_request(content_brief)
        
        logger.info(f"Generating content for topic: {content_brief.topic}")
        
        try:
            # Create the messages array
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_request}
            ]
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4000
            }
            
            # Define headers with API key
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make the API call directly with requests
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            # Raise an exception if the request failed
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Extract the content from the response
            generated_text = response_data["choices"][0]["message"]["content"]
        except Exception as e:
            import traceback
            print(f"Error in generate_content_draft: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to generate content draft: {e}")
        
        # Create the generated content object
        generated_content = GeneratedContent(
            content=generated_text,
            content_brief=content_brief,
            style_parameters=style_parameters if isinstance(style_parameters, StyleParameters) 
                            else StyleParameters(**style_parameters),
            metadata={
                "model": self.model,
                "temperature": 0.7,
                "generation_date": str(Path().absolute())
            }
        )
        
        return {
            "content": generated_text,
            "full_response": generated_content.model_dump()
        }
    
    def _create_style_replication_prompt(
        self,
        persona: Dict[str, Any],
        style_parameters: StyleParameters
    ) -> str:
        """
        Create a detailed system prompt for style replication.
        
        Args:
            persona: The author persona
            style_parameters: Parameters controlling style adherence
            
        Returns:
            A system prompt string
        """
        # Start with base prompt
        prompt = f"""You are replicating the writing style of {persona.get('name', 'the author')}. 
        Your task is to generate content that authentically captures this author's unique voice and style.
        
        """
        
        # Add writing voice summary if available
        if 'writing_voice_summary' in persona:
            prompt += f"WRITING VOICE SUMMARY:\n{persona['writing_voice_summary']}\n\n"
        elif 'system_message' in persona:
            prompt += f"PERSONA DESCRIPTION:\n{persona['system_message']}\n\n"
            
        # Add style parameters guidance
        prompt += "STYLE REPLICATION PARAMETERS:\n"
        
        # Base fidelity level
        prompt += f"- Overall style fidelity: {style_parameters.style_fidelity * 100:.0f}% adherence to the author's style\n"
        
        # Add specific guidance based on style parameters
        if style_parameters.vocabulary_adherence > 0.8:
            prompt += "- Use vocabulary very similar to the author's typical word choices\n"
        elif style_parameters.vocabulary_adherence > 0.5:
            prompt += "- Use vocabulary somewhat similar to the author's, with some flexibility\n"
        else:
            prompt += "- Use vocabulary loosely inspired by the author's, with significant flexibility\n"
            
        if style_parameters.sentence_structure_adherence > 0.8:
            prompt += "- Closely match the author's typical sentence structures and lengths\n"
        elif style_parameters.sentence_structure_adherence > 0.5:
            prompt += "- Somewhat match the author's sentence patterns, with some variation\n"
        else:
            prompt += "- Use sentence structures loosely inspired by the author's, with significant flexibility\n"
            
        if style_parameters.rhetorical_devices_usage > 0.8:
            prompt += "- Frequently incorporate the author's characteristic rhetorical devices\n"
        elif style_parameters.rhetorical_devices_usage > 0.5:
            prompt += "- Occasionally use the author's rhetorical devices where appropriate\n"
        else:
            prompt += "- Sparingly use the author's rhetorical devices, focusing more on content\n"
            
        if style_parameters.tone_consistency > 0.8:
            prompt += "- Maintain a tone very consistent with the author's typical expression\n"
        elif style_parameters.tone_consistency > 0.5:
            prompt += "- Aim for a tone generally consistent with the author's, with some adaptation\n"
        else:
            prompt += "- Adapt tone substantially to the content requirements while keeping subtle author cues\n"
            
        if style_parameters.quirk_frequency > 0.8:
            prompt += "- Frequently incorporate the author's writing quirks and idiosyncrasies\n"
        elif style_parameters.quirk_frequency > 0.5:
            prompt += "- Occasionally include the author's writing quirks where natural\n"
        else:
            prompt += "- Minimize the author's writing quirks, focusing on smoother content\n"
            
        if style_parameters.creative_freedom < 0.3:
            prompt += "- Stay very close to the author's established patterns with minimal deviation\n"
        elif style_parameters.creative_freedom < 0.7:
            prompt += "- Allow moderate creative adaptation while keeping the author's core style\n"
        else:
            prompt += "- Exercise significant creative freedom while maintaining the essence of the author's style\n"
            
        # Add style markers section if available
        if 'style_markers' in persona:
            prompt += "\nDISTINCTIVE STYLE MARKERS TO INCORPORATE:\n"
            
            style_markers = persona['style_markers']
            
            # Add signature phrases
            if 'signature_phrases' in style_markers and style_markers['signature_phrases']:
                phrases = style_markers['signature_phrases']
                if isinstance(phrases, list) and phrases:
                    prompt += f"- Signature phrases: {', '.join(phrases[:5])}\n"
                    
            # Add punctuation patterns
            if 'punctuation_patterns' in style_markers:
                punct = style_markers['punctuation_patterns']
                if isinstance(punct, dict) and punct:
                    punct_str = ", ".join(f"{k}: {v}" for k, v in list(punct.items())[:3])
                    prompt += f"- Punctuation patterns: {punct_str}\n"
                    
            # Add structural quirks
            if 'structural_quirks' in style_markers and style_markers['structural_quirks']:
                quirks = style_markers['structural_quirks']
                if isinstance(quirks, list) and quirks:
                    prompt += f"- Structural quirks: {', '.join(quirks[:5])}\n"
        
        # Add a reminder about balancing style and content
        prompt += "\nIMPORTANT: While replicating the author's style is important, the content must still effectively address the topic and purpose described in the user's request. Balance style authenticity with content effectiveness."
        
        return prompt
    
    def _create_content_request(self, content_brief: ContentBrief) -> str:
        """
        Create a user prompt for content generation based on the brief.
        
        Args:
            content_brief: The content brief
            
        Returns:
            A user prompt string
        """
        request = f"""Please write {content_brief.content_type} about {content_brief.topic}."""
        
        if content_brief.target_audience and content_brief.target_audience != "General":
            request += f" The target audience is {content_brief.target_audience}."
            
        if content_brief.length:
            request += f" It should be approximately {content_brief.length} in length."
            
        if content_brief.tone:
            request += f" The tone should be {content_brief.tone}."
            
        if content_brief.key_points:
            request += "\n\nPlease include the following key points:\n"
            for point in content_brief.key_points:
                request += f"- {point}\n"
                
        if content_brief.additional_requirements:
            request += "\n\nAdditional requirements:\n"
            for key, value in content_brief.additional_requirements.items():
                request += f"- {key}: {value}\n"
                
        return request
    
    def _refine_content(
        self,
        persona: Dict[str, Any],
        content: str,
        feedback: Dict[str, Any],
        style_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Refine content based on feedback.
        
        Args:
            persona: The author persona
            content: The content to refine
            feedback: Feedback on the content
            style_parameters: Optional style parameters to control generation
            
        Returns:
            A dictionary with the refined content
        """
        import requests
        
        # Convert dictionaries to models if needed
        if isinstance(feedback, dict):
            feedback = GenerationFeedback(**feedback)
            
        if style_parameters is None:
            style_parameters = StyleParameters().model_dump()
        elif isinstance(style_parameters, dict):
            style_parameters = StyleParameters(**style_parameters)
            
        # Adjust style parameters based on feedback
        adjusted_parameters = self._adjust_style_parameters(style_parameters, feedback)
            
        # Prepare the system prompt based on the persona and adjusted style parameters
        system_prompt = self._create_style_replication_prompt(persona, adjusted_parameters)
        
        # Prepare the refinement request
        refinement_request = self._create_refinement_request(content, feedback)
        
        logger.info(f"Refining content based on feedback (overall rating: {feedback.overall_rating}/5)")
        
        try:
            # Create the messages array
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original content:\n\n{content}"},
                {"role": "user", "content": refinement_request}
            ]
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4000
            }
            
            # Define headers with API key
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make the API call directly with requests
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            # Raise an exception if the request failed
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Extract the content from the response
            refined_text = response_data["choices"][0]["message"]["content"]
        except Exception as e:
            import traceback
            print(f"Error in refine_content: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to refine content: {e}")
        
        return {
            "content": refined_text,
            "adjusted_parameters": adjusted_parameters.model_dump()
        }
    
    def _adjust_style_parameters(
        self,
        style_parameters: StyleParameters,
        feedback: GenerationFeedback
    ) -> StyleParameters:
        """
        Adjust style parameters based on feedback.
        
        Args:
            style_parameters: Current style parameters
            feedback: Feedback on the content
            
        Returns:
            Adjusted style parameters
        """
        # Create a copy of the parameters to modify
        params = style_parameters.model_copy()
        
        # Adjust style fidelity based on style match rating
        if feedback.style_match_rating < 3:
            # Increase style fidelity if rating is low
            params.style_fidelity = min(1.0, params.style_fidelity + 0.1)
        elif feedback.style_match_rating >= 4:
            # Slightly decrease if already very good (for balance with content)
            params.style_fidelity = max(0.5, params.style_fidelity - 0.05)
            
        # Adjust based on elements to emphasize
        for element in feedback.elements_to_emphasize:
            element_lower = element.lower()
            
            if "vocabulary" in element_lower or "word choice" in element_lower:
                params.vocabulary_adherence = min(1.0, params.vocabulary_adherence + 0.1)
                
            if "sentence" in element_lower or "structure" in element_lower:
                params.sentence_structure_adherence = min(1.0, params.sentence_structure_adherence + 0.1)
                
            if "rhetorical" in element_lower or "device" in element_lower or "figure" in element_lower:
                params.rhetorical_devices_usage = min(1.0, params.rhetorical_devices_usage + 0.1)
                
            if "tone" in element_lower or "voice" in element_lower:
                params.tone_consistency = min(1.0, params.tone_consistency + 0.1)
                
            if "quirk" in element_lower or "idiosyncras" in element_lower or "unique" in element_lower:
                params.quirk_frequency = min(1.0, params.quirk_frequency + 0.1)
                
        # Adjust based on elements to reduce
        for element in feedback.elements_to_reduce:
            element_lower = element.lower()
            
            if "vocabulary" in element_lower or "word choice" in element_lower:
                params.vocabulary_adherence = max(0.0, params.vocabulary_adherence - 0.1)
                
            if "sentence" in element_lower or "structure" in element_lower:
                params.sentence_structure_adherence = max(0.0, params.sentence_structure_adherence - 0.1)
                
            if "rhetorical" in element_lower or "device" in element_lower or "figure" in element_lower:
                params.rhetorical_devices_usage = max(0.0, params.rhetorical_devices_usage - 0.1)
                
            if "tone" in element_lower or "voice" in element_lower:
                params.tone_consistency = max(0.0, params.tone_consistency - 0.1)
                
            if "quirk" in element_lower or "idiosyncras" in element_lower or "unique" in element_lower:
                params.quirk_frequency = max(0.0, params.quirk_frequency - 0.1)
                
        # If content quality is low but style match is good, increase creative freedom
        if feedback.content_quality_rating < 3 and feedback.style_match_rating >= 4:
            params.creative_freedom = min(1.0, params.creative_freedom + 0.15)
            
        return params
    
    def _create_refinement_request(self, content: str, feedback: GenerationFeedback) -> str:
        """
        Create a refinement request based on feedback.
        
        Args:
            content: The content to refine
            feedback: Feedback on the content
            
        Returns:
            A refinement request string
        """
        request = f"""Please refine the above content based on the following feedback:

Overall Rating: {feedback.overall_rating}/5
Style Match Rating: {feedback.style_match_rating}/5
Content Quality Rating: {feedback.content_quality_rating}/5

"""
        
        if feedback.specific_feedback:
            request += "Specific feedback to address:\n"
            for point in feedback.specific_feedback:
                request += f"- {point}\n"
            request += "\n"
            
        if feedback.elements_to_emphasize:
            request += "Stylistic elements to emphasize more:\n"
            for element in feedback.elements_to_emphasize:
                request += f"- {element}\n"
            request += "\n"
            
        if feedback.elements_to_reduce:
            request += "Stylistic elements to tone down:\n"
            for element in feedback.elements_to_reduce:
                request += f"- {element}\n"
            request += "\n"
            
        request += """
Please provide a revised version that addresses this feedback while maintaining the core content and purpose.
Ensure the refinements make the content more effective while better matching the author's style as specified.
"""
        
        return request
    
    def _analyze_style_adherence(
        self,
        persona: Dict[str, Any],
        content: str
    ) -> Dict[str, Any]:
        """
        Analyze how well content adheres to the author's style.
        
        Args:
            persona: The author persona
            content: The content to analyze
            
        Returns:
            A dictionary with style adherence analysis
        """
        import requests
        
        # Prepare the system prompt
        system_prompt = f"""You are an expert literary analyst specializing in authorial style. 
        
        Your task is to analyze how well the provided content matches the style of {persona.get('name', 'the author')}.
        
        Evaluate the following aspects of style adherence:
        1. Vocabulary and word choice patterns
        2. Sentence structures and lengths
        3. Rhetorical devices and figurative language
        4. Tone and voice consistency
        5. Distinctive quirks and idiosyncrasies
        
        For each aspect, provide:
        - A rating from 1-10
        - Specific examples from the text
        - Suggestions for improvement
        
        Conclude with an overall style adherence score and summary assessment.
        """
        
        # Add author style information if available
        if 'writing_voice_summary' in persona:
            system_prompt += f"\n\nThe author's writing style is characterized as follows:\n{persona['writing_voice_summary']}"
        elif 'system_message' in persona:
            system_prompt += f"\n\nThe author's persona is described as follows:\n{persona['system_message']}"
            
        logger.info(f"Analyzing style adherence for content (length: {len(content)} chars)")
        
        try:
            # Create the messages array
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please analyze this content for style adherence:\n\n{content}"}
            ]
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.5,
                "response_format": {"type": "json_object"}
            }
            
            # Define headers with API key
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make the API call directly with requests
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            # Raise an exception if the request failed
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Extract the content from the response and parse as JSON
            analysis = json.loads(response_data["choices"][0]["message"]["content"])
            
            return analysis
        except Exception as e:
            import traceback
            print(f"Error in analyze_style_adherence: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to analyze style adherence: {e}")
    
    def generate_content(
        self,
        persona: Any,
        content_brief: Dict[str, Any],
        style_parameters: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        format: str = 'json'
    ) -> GeneratedContent:
        """
        Generate content in the author's style based on a content brief.
        
        Args:
            persona: The author persona (AuthorPersona, Persona, or dictionary)
            content_brief: The content brief as a dictionary
            style_parameters: Optional style parameters to control generation
            output_path: Optional path to save the generated content
            format: Format to save in ('json', 'txt', 'md', 'html')
            
        Returns:
            A GeneratedContent object
        """
        # Prepare the persona context
        persona_context = self._prepare_persona_context(persona)
        
        # Convert dictionaries to models if needed
        if isinstance(content_brief, dict):
            content_brief = ContentBrief(**content_brief)
            
        if style_parameters is None:
            style_parameters = StyleParameters().model_dump()
        
        logger.info(f"Generating content for topic: {content_brief.topic}")
        
        # Generate the content
        result = self._generate_content_draft(
            persona=persona_context,
            content_brief=content_brief,
            style_parameters=style_parameters
        )
        
        # Create the generated content object
        if isinstance(result, dict) and 'full_response' in result:
            # Already has the full GeneratedContent model
            generated_content = GeneratedContent(**result['full_response'])
        else:
            # Create a new GeneratedContent model
            content_text = result['content'] if isinstance(result, dict) else result
            
            generated_content = GeneratedContent(
                content=content_text,
                content_brief=content_brief,
                style_parameters=StyleParameters(**style_parameters) if isinstance(style_parameters, dict) 
                                else style_parameters,
                metadata={
                    "model": self.model,
                    "persona_name": persona_context.get('name', 'Unknown Author'),
                    "generation_date": str(Path().absolute())
                }
            )
        
        # Save the content if requested
        if output_path:
            self.save_generated_content(generated_content, output_path, format)
            
        return generated_content
    
    def refine_content_with_feedback(
        self,
        persona: Any,
        content: Union[str, GeneratedContent],
        feedback: Dict[str, Any],
        style_parameters: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        format: str = 'json'
    ) -> GeneratedContent:
        """
        Refine content based on feedback.
        
        Args:
            persona: The author persona (AuthorPersona, Persona, or dictionary)
            content: The content to refine, either as a string or GeneratedContent object
            feedback: Feedback on the content as a dictionary
            style_parameters: Optional style parameters to control generation
            output_path: Optional path to save the refined content
            format: Format to save in ('json', 'txt', 'md', 'html')
            
        Returns:
            A GeneratedContent object with the refined content
        """
        # Prepare the persona context
        persona_context = self._prepare_persona_context(persona)
        
        # Extract content string and history
        content_text = ""
        content_brief = None
        current_style_parameters = None
        refinement_history = []
        
        # Check if we're dealing with a GeneratedContent object or a string
        if isinstance(content, GeneratedContent):
            content_text = content.content
            content_brief = content.content_brief
            current_style_parameters = content.style_parameters
            refinement_history = content.refinement_history.copy()
        else:
            content_text = content
            
        # Convert feedback to model if needed
        if isinstance(feedback, dict):
            feedback = GenerationFeedback(**feedback)
            
        # If no style parameters provided, use the ones from the content object or defaults
        if style_parameters is None:
            if current_style_parameters is not None:
                style_parameters = current_style_parameters.model_dump()
            else:
                style_parameters = StyleParameters().model_dump()
        
        logger.info(f"Refining content based on feedback")
        
        # Refine the content
        result = self._refine_content(
            persona=persona_context,
            content=content_text,
            feedback=feedback.model_dump(),
            style_parameters=style_parameters
        )
        
        refined_text = result['content']
        adjusted_parameters = result.get('adjusted_parameters', style_parameters)
        
        # Record this refinement in history
        refinement_record = {
            "version": len(refinement_history) + 1,
            "feedback": feedback.model_dump(),
            "previous_content": content_text[:100] + "..." if len(content_text) > 100 else content_text,
            "style_parameters_before": style_parameters,
            "style_parameters_after": adjusted_parameters,
            "timestamp": str(Path().absolute())
        }
        
        refinement_history.append(refinement_record)
        
        # Create or update the generated content object
        if content_brief is None:
            # Create a minimal content brief if none exists
            content_brief = ContentBrief(
                topic="Unknown Topic",
                content_type="Unknown Type"
            )
            
        generated_content = GeneratedContent(
            content=refined_text,
            content_brief=content_brief,
            style_parameters=StyleParameters(**adjusted_parameters),
            refinement_history=refinement_history,
            metadata={
                "model": self.model,
                "persona_name": persona_context.get('name', 'Unknown Author'),
                "generation_date": str(Path().absolute()),
                "refinement_count": len(refinement_history)
            }
        )
        
        # Save the content if requested
        if output_path:
            self.save_generated_content(generated_content, output_path, format)
            
        return generated_content
    
    def save_generated_content(
        self,
        generated_content: GeneratedContent,
        output_path: str,
        format: str = 'json'
    ) -> str:
        """
        Save generated content to a file.
        
        Args:
            generated_content: The generated content to save
            output_path: Path to save the content to
            format: Format to save in ('json', 'txt', 'md', 'html')
            
        Returns:
            The path to the saved file
        """
        # Ensure the directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        format = format.lower()
        
        # Save based on format
        if format == 'json':
            # Save full object as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(generated_content.model_dump(), f, indent=2)
        elif format in ['txt', 'md', 'markdown']:
            # Save just the content text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(generated_content.content)
        elif format == 'html':
            # Save as HTML with metadata
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{generated_content.content_brief.topic}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .content {{ margin-top: 30px; }}
        .metadata {{ font-size: 0.8em; color: #666; border-top: 1px solid #ccc; margin-top: 30px; padding-top: 10px; }}
    </style>
</head>
<body>
    <div class="content">
        {generated_content.content.replace('\n', '<br>')}
    </div>
    <div class="metadata">
        <p>Generated in the style of: {generated_content.metadata.get('persona_name', 'Unknown Author')}</p>
        <p>Topic: {generated_content.content_brief.topic}</p>
        <p>Content Type: {generated_content.content_brief.content_type}</p>
        <p>Generated: {generated_content.metadata.get('generation_date', 'Unknown')}</p>
    </div>
</body>
</html>"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        else:
            # Default to txt if format is not recognized
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(generated_content.content)
                
        logger.info(f"Content saved to {output_path}")
        return output_path
