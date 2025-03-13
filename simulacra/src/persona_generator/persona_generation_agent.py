import os
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from ..openai_agents import Agent, AgentTool, NamedAgentTool, AgentAction
from pydantic import BaseModel, Field

from ..file_operations.persona_serializer import PersonaSerializer

from ..document_analysis.document_analyzer import AuthorAnalysis, DocumentFeatures, PsychologicalProfile
from .persona import Persona


class WritingCharacteristics(BaseModel):
    """Detailed profile of an author's writing characteristics."""
    
    vocabulary_profile: Dict[str, Any] = Field(
        description="Vocabulary richness, complexity, and unique word usage patterns"
    )
    
    sentence_construction: Dict[str, Any] = Field(
        description="Sentence length, complexity, and structural preferences"
    )
    
    rhetorical_devices: List[str] = Field(
        description="Commonly used rhetorical devices and figurative language",
        default_factory=list
    )
    
    tone_patterns: Dict[str, float] = Field(
        description="Distribution of tones used in writing (formal, casual, etc.)",
        default_factory=dict
    )
    
    organizational_patterns: List[str] = Field(
        description="How the author typically organizes content and arguments",
        default_factory=list
    )


class StyleMarkers(BaseModel):
    """Unique stylistic markers that distinguish the author's writing."""
    
    signature_phrases: List[str] = Field(
        description="Recurring phrases or expressions unique to the author",
        default_factory=list
    )
    
    punctuation_patterns: Dict[str, Any] = Field(
        description="Distinctive use of punctuation",
        default_factory=dict
    )
    
    transition_preferences: List[str] = Field(
        description="Preferred transition words or techniques",
        default_factory=list
    )
    
    structural_quirks: List[str] = Field(
        description="Unusual structural choices that appear consistently",
        default_factory=list
    )
    
    lexical_preferences: Dict[str, Any] = Field(
        description="Word choice patterns and preferences",
        default_factory=dict
    )


class PsychologicalTraits(BaseModel):
    """Psychological traits reflected in the author's writing."""
    
    personality_dimensions: Dict[str, float] = Field(
        description="Big Five personality traits and other personality dimensions",
        default_factory=dict
    )
    
    cognitive_style: Dict[str, Any] = Field(
        description="Information processing and problem-solving approaches",
        default_factory=dict
    )
    
    emotional_patterns: Dict[str, float] = Field(
        description="Emotional expression tendencies",
        default_factory=dict
    )
    
    values_indicators: List[str] = Field(
        description="Values and beliefs that emerge in writing",
        default_factory=list
    )
    
    social_orientation: Dict[str, Any] = Field(
        description="Social interaction preferences and communication style",
        default_factory=dict
    )


class AuthorPersona(BaseModel):
    """Comprehensive persona model of an author based on their writing."""
    
    name: str = Field(description="Generated name for the author persona")
    
    writing_characteristics: WritingCharacteristics = Field(
        description="Detailed profile of writing characteristics"
    )
    
    style_markers: StyleMarkers = Field(
        description="Unique stylistic markers"
    )
    
    psychological_traits: PsychologicalTraits = Field(
        description="Psychological traits reflected in the writing"
    )
    
    writing_voice_summary: str = Field(
        description="Concise summary of the author's writing voice"
    )
    
    recommended_topics: List[str] = Field(
        description="Topics that would align well with this author's style and interests",
        default_factory=list
    )
    
    author_background: Dict[str, Any] = Field(
        description="Inferred background details that might explain writing patterns",
        default_factory=dict
    )


class PersonaGenerationAgent:
    """
    An agent that takes document analysis results and generates a comprehensive persona model
    of the author. Uses the OpenAI Agents SDK for enhanced reasoning capabilities.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "gpt-4-turbo"):
        """
        Initialize the persona generation agent.
        
        Args:
            api_key: OpenAI API key. If None, it will be loaded from environment variables.
            model: The model to use for persona generation.
        """
        self.api_key = api_key
        self.model = model
        
        # Initialize the agent with tools
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """Create an OpenAI Agent with the necessary tools for persona generation."""
        
        tools = [
            NamedAgentTool(
                name="extract_writing_characteristics",
                description="Extract detailed writing characteristics from document analysis",
                callable=self._extract_writing_characteristics,
            ),
            NamedAgentTool(
                name="identify_style_markers",
                description="Identify unique stylistic markers from document analysis",
                callable=self._identify_style_markers,
            ),
            NamedAgentTool(
                name="analyze_psychological_traits",
                description="Analyze psychological traits reflected in the writing",
                callable=self._analyze_psychological_traits,
            ),
            NamedAgentTool(
                name="generate_author_background",
                description="Generate a plausible author background based on writing analysis",
                callable=self._generate_author_background,
            ),
            NamedAgentTool(
                name="suggest_topics",
                description="Suggest topics that would suit this author's style",
                callable=self._suggest_topics,
            ),
        ]
        
        system_prompt = """You are an expert persona generation agent specializing in creating comprehensive 
        author models based on writing analysis. Your task is to take document analysis results and generate 
        a detailed persona that captures the essence of the author's writing style, unique markers, and 
        psychological traits.
        
        Your persona generation should be evidence-based, drawing directly from the analysis data. Focus on 
        creating a cohesive and realistic author model that could be used to replicate the writing style.
        
        You have tools available to:
        1. Extract detailed writing characteristics
        2. Identify unique stylistic markers
        3. Analyze psychological traits reflected in the writing
        4. Generate a plausible author background
        5. Suggest topics that would suit this author's style
        
        Create personas that are nuanced, distinctive, and based firmly on the provided analysis.
        """
        
        return Agent(
            system_prompt=system_prompt,
            tools=tools,
            model=self.model,
            api_key=self.api_key
        )
    
    def _extract_writing_characteristics(self, analysis_json: str) -> Dict[str, Any]:
        """
        Extract detailed writing characteristics from document analysis.
        
        Args:
            analysis_json: JSON string of the document analysis
            
        Returns:
            A dictionary of writing characteristics
        """
        from openai import OpenAI
        # Initialize client with only the API key to avoid 'proxies' parameter issue
        client = OpenAI(api_key=self.api_key)
        
        system_prompt = """You are an expert writing analyst. Extract detailed writing characteristics from 
        the provided document analysis. Focus on:
        
        1. Vocabulary profile (richness, complexity, unique word usage)
        2. Sentence construction patterns (length, complexity, structure)
        3. Rhetorical devices commonly employed
        4. Tone patterns and distribution
        5. Organizational and structural patterns
        
        Provide specific examples from the analysis where possible. Your response should be structured in JSON format.
        """
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_json}
            ],
            response_format={"type": "json_object"}
        )
        
        characteristics = json.loads(response.choices[0].message.content)
        return characteristics
    
    def _identify_style_markers(self, analysis_json: str) -> Dict[str, Any]:
        """
        Identify unique stylistic markers from document analysis.
        
        Args:
            analysis_json: JSON string of the document analysis
            
        Returns:
            A dictionary of stylistic markers
        """
        from openai import OpenAI
        # Initialize client with only the API key to avoid 'proxies' parameter issue
        client = OpenAI(api_key=self.api_key)
        
        system_prompt = """You are an expert in authorial fingerprinting. Identify the unique stylistic markers 
        that distinguish this author from others based on the provided document analysis. Focus on:
        
        1. Signature phrases or expressions
        2. Distinctive punctuation patterns
        3. Transition preferences
        4. Structural quirks
        5. Lexical preferences and word choice patterns
        
        Identify patterns that would help distinguish this author's writing from others. Your response 
        should be structured in JSON format.
        """
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_json}
            ],
            response_format={"type": "json_object"}
        )
        
        style_markers = json.loads(response.choices[0].message.content)
        return style_markers
    
    def _analyze_psychological_traits(self, analysis_json: str) -> Dict[str, Any]:
        """
        Analyze psychological traits reflected in the writing.
        
        Args:
            analysis_json: JSON string of the document analysis
            
        Returns:
            A dictionary of psychological traits
        """
        from openai import OpenAI
        # Initialize client with only the API key to avoid 'proxies' parameter issue
        client = OpenAI(api_key=self.api_key)
        
        system_prompt = """You are an expert in psycholinguistics. Analyze the psychological traits reflected 
        in the author's writing based on the provided document analysis. Focus on:
        
        1. Personality dimensions (Big Five traits and others)
        2. Cognitive style (information processing, problem-solving approaches)
        3. Emotional patterns and expression
        4. Values and beliefs that emerge in writing
        5. Social orientation and communication preferences
        
        Be nuanced in your analysis and avoid overstating confidence in psychological inferences. 
        Your response should be structured in JSON format.
        """
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_json}
            ],
            response_format={"type": "json_object"}
        )
        
        psychological_traits = json.loads(response.choices[0].message.content)
        return psychological_traits
    
    def _generate_author_background(self, analysis_json: str) -> Dict[str, Any]:
        """
        Generate a plausible author background based on writing analysis.
        
        Args:
            analysis_json: JSON string of the document analysis
            
        Returns:
            A dictionary with background information
        """
        from openai import OpenAI
        # Initialize client with only the API key to avoid 'proxies' parameter issue
        client = OpenAI(api_key=self.api_key)
        
        system_prompt = """You are an expert in author profiling. Based on the provided document analysis, 
        generate a plausible background for the author that might explain their writing patterns. Include:
        
        1. Potential educational background
        2. Likely professional experience
        3. Possible cultural influences
        4. Inferred intellectual interests
        5. Communication and social preferences
        
        Make sure your inferences are reasonably supported by the analysis data. Acknowledge uncertainty 
        where appropriate. Your response should be structured in JSON format.
        """
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_json}
            ],
            response_format={"type": "json_object"}
        )
        
        background = json.loads(response.choices[0].message.content)
        return background
    
    def _suggest_topics(self, analysis_json: str) -> List[str]:
        """
        Suggest topics that would suit this author's style.
        
        Args:
            analysis_json: JSON string of the document analysis
            
        Returns:
            A list of suggested topics
        """
        from openai import OpenAI
        # Initialize client with only the API key to avoid 'proxies' parameter issue
        client = OpenAI(api_key=self.api_key)
        
        system_prompt = """You are an expert content strategist. Based on the provided document analysis, 
        suggest 5-10 topics or subjects that would align well with this author's style, interests, and strengths.
        
        Consider the author's vocabulary, tone, structural preferences, and apparent areas of interest or expertise.
        Provide brief reasoning for why each topic would be suitable.
        
        Your response should be structured in JSON format as an array of objects, each with a "topic" and "rationale" field.
        """
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_json}
            ],
            response_format={"type": "json_object"}
        )
        
        topics_data = json.loads(response.choices[0].message.content)
        
        # Extract just the topic names for the return value
        if isinstance(topics_data, list):
            topics = [item.get("topic", "") for item in topics_data if "topic" in item]
        elif "topics" in topics_data:
            topics = topics_data["topics"]
        else:
            topics = []
            
        return topics
    
    def _generate_name(self, author_data: Dict[str, Any]) -> str:
        """
        Generate a fitting name for the author persona based on analysis data.
        
        Args:
            author_data: Combined data about the author
            
        Returns:
            A generated name
        """
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        
        system_prompt = """You are a creative naming specialist. Based on the provided author analysis data, 
        generate a fitting name for this author persona. The name should reflect aspects of their writing style, 
        psychological traits, or thematic tendencies.
        
        Provide just the name without explanation.
        """
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(author_data)}
            ]
        )
        
        name = response.choices[0].message.content.strip()
        return name
    
    def _generate_writing_voice_summary(self, author_data: Dict[str, Any]) -> str:
        """
        Generate a concise summary of the author's writing voice.
        
        Args:
            author_data: Combined data about the author
            
        Returns:
            A summary of the author's writing voice
        """
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        
        system_prompt = """You are an expert writing analyst. Based on the provided author analysis data, 
        generate a concise (2-3 paragraph) summary of the author's writing voice. 
        
        Focus on what makes their writing distinctive and memorable. Synthesize information about their 
        vocabulary, sentence structure, stylistic markers, and psychological traits into a cohesive description.
        """
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(author_data)}
            ]
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
    
    def generate_persona_from_analysis(self, analysis: AuthorAnalysis) -> AuthorPersona:
        """
        Generate a comprehensive author persona from document analysis results.
        
        Args:
            analysis: The complete author analysis from DocumentAnalysisAgent
            
        Returns:
            An AuthorPersona object
        """
        # Convert the analysis to JSON for processing
        analysis_json = json.dumps(analysis.model_dump())
        
        print("Extracting writing characteristics...")
        writing_characteristics_data = self._extract_writing_characteristics(analysis_json)
        
        print("Identifying style markers...")
        style_markers_data = self._identify_style_markers(analysis_json)
        
        print("Analyzing psychological traits...")
        psychological_traits_data = self._analyze_psychological_traits(analysis_json)
        
        print("Generating author background...")
        author_background = self._generate_author_background(analysis_json)
        
        print("Suggesting suitable topics...")
        recommended_topics = self._suggest_topics(analysis_json)
        
        # Combine all data for generating name and voice summary
        author_data = {
            "writing_characteristics": writing_characteristics_data,
            "style_markers": style_markers_data,
            "psychological_traits": psychological_traits_data,
            "background": author_background
        }
        
        print("Generating author name...")
        name = self._generate_name(author_data)
        
        print("Generating writing voice summary...")
        writing_voice_summary = self._generate_writing_voice_summary(author_data)
        
        # Create the author persona
        author_persona = AuthorPersona(
            name=name,
            writing_characteristics=WritingCharacteristics(**writing_characteristics_data),
            style_markers=StyleMarkers(**style_markers_data),
            psychological_traits=PsychologicalTraits(**psychological_traits_data),
            writing_voice_summary=writing_voice_summary,
            recommended_topics=recommended_topics,
            author_background=author_background
        )
        
        return author_persona
    
    def save_persona(self, persona: AuthorPersona, output_path: str, format: str = 'json') -> str:
        """
        Save the author persona to a file.
        
        Args:
            persona: The author persona to save
            output_path: Path to save the persona to
            format: Format to save in ('json', 'yaml', 'markdown', 'md', 'txt')
            
        Returns:
            The path to the saved file
        """
        # Ensure the directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Convert the persona to a dictionary
        persona_dict = persona.model_dump()
        
        # Save as JSON (default format)
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(persona_dict, f, indent=2)
        else:
            # For other formats, use PersonaSerializer after converting to a Simulacra Persona
            simulacra_persona = self.convert_to_simulacra_persona(persona)
            serializer = PersonaSerializer()
            result = serializer.save_persona(
                persona=simulacra_persona,
                output_path=output_path,
                format=format,
                overwrite=True
            )
            
            if not result.success:
                raise ValueError(f"Failed to save persona: {result.metadata.get('error', 'Unknown error')}")
            
        return output_path
    
    def generate_from_documents(self, folder_path: str, file_extensions: List[str]) -> AuthorPersona:
        """
        Generate a comprehensive author persona from a folder of document files.
        
        Args:
            folder_path: Path to the folder containing documents
            file_extensions: List of file extensions to include (e.g., ['txt', 'md', 'docx'])
            
        Returns:
            An AuthorPersona object
        """
        from pathlib import Path
        import json
        
        # Import document analysis functionality
        from ..document_analysis.document_analyzer import DocumentAnalyzer, AnalysisRequest
        
        print(f"Analyzing documents in {folder_path} with extensions: {file_extensions}")
        
        # Initialize document analyzer
        analyzer = DocumentAnalyzer(api_key=self.api_key)
        
        # Create analysis request
        request = AnalysisRequest(
            content_paths=[folder_path],
            file_extensions=file_extensions,
            analysis_type="author",
            include_psychological_profile=True,
            include_document_features=True
        )
        
        # Run the analysis
        analysis_result = analyzer.analyze(request)
        
        # Generate persona from the analysis result
        print("Generating persona from analysis results...")
        author_persona = self.generate_persona_from_analysis(analysis_result.author_analysis)
        
        return author_persona
    
    def convert_to_simulacra_persona(self, author_persona: AuthorPersona) -> Persona:
        """
        Convert an AuthorPersona to a Simulacra Persona object.
        
        Args:
            author_persona: The author persona to convert
            
        Returns:
            A Simulacra Persona object
        """
        # Extract personality traits from the psychological profile
        traits = []
        
        # Add traits based on Big Five dimensions
        personality = author_persona.psychological_traits.personality_dimensions
        for trait, value in personality.items():
            if isinstance(value, (int, float)) and value >= 0.7:
                traits.append(f"high {trait}")
            elif isinstance(value, (int, float)) and value <= 0.3:
                traits.append(f"low {trait}")
                
        # Add writing style traits
        for marker in author_persona.style_markers.structural_quirks:
            traits.append(marker)
            
        # Limit to most important traits if we have too many
        if len(traits) > 10:
            traits = traits[:10]
            
        # Generate a background from the author background data
        background_parts = []
        for key, value in author_persona.author_background.items():
            if isinstance(value, str):
                background_parts.append(f"{key}: {value}")
            elif isinstance(value, (list, tuple)):
                background_parts.append(f"{key}: {', '.join(value)}")
                
        background = " ".join(background_parts)
        
        # Determine communication style from psychological traits
        communication_style = "Formal and structured"  # Default
        if "social_orientation" in author_persona.psychological_traits.model_dump():
            social = author_persona.psychological_traits.social_orientation
            if "communication_preferences" in social:
                prefs = social["communication_preferences"]
                if isinstance(prefs, list) and prefs:
                    communication_style = ", ".join(prefs[:3])
                elif isinstance(prefs, str):
                    communication_style = prefs
        
        # Determine knowledge areas from recommended topics
        knowledge_areas = author_persona.recommended_topics[:5] if author_persona.recommended_topics else []
        
        # Create the persona
        simulacra_persona = Persona(
            name=author_persona.name,
            traits=traits,
            background=background,
            communication_style=communication_style,
            knowledge_areas=knowledge_areas,
            additional_details={
                "writing_voice_summary": author_persona.writing_voice_summary,
                "vocabulary_profile": author_persona.writing_characteristics.vocabulary_profile,
                "rhetorical_devices": author_persona.writing_characteristics.rhetorical_devices
            }
        )
        
        return simulacra_persona
