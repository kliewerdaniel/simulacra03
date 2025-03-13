import os
from typing import Dict, List, Any, Optional, Union, Set
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Use local stub implementation instead of the package
from ..openai_agents import Agent, AgentTool, NamedAgentTool, AgentAction
from pydantic import BaseModel, Field

from ..file_operations.directory_traversal import DirectoryTraverser, FileMetadata
from ..file_operations.document_parser import DocumentParser, DocumentContent
from .openai_client_wrapper import OpenAIClientWrapper

class DocumentFeatures(BaseModel):
    """Model for storing document features extracted during analysis."""
    
    # Vocabulary statistics
    vocabulary_size: int
    average_word_length: float
    word_frequencies: Dict[str, int]
    rare_words: List[str]
    
    # Sentence structure
    average_sentence_length: float
    sentence_length_variation: float
    sentence_structures: Dict[str, int]  # e.g., "simple", "compound", "complex", etc.
    
    # Stylistic elements
    idioms: List[str]
    metaphors: List[str]
    transition_phrases: List[str]
    
    # Writing patterns
    paragraph_structure: Dict[str, Any]
    punctuation_usage: Dict[str, int]
    passive_voice_frequency: float
    active_voice_frequency: float
    
    # Meta statistics
    document_count: int
    total_word_count: int
    total_sentence_count: int
    
class PsychologicalProfile(BaseModel):
    """Model for psychological traits inferred from writing style."""
    
    # Big Five personality traits (estimated from writing style)
    openness: float = Field(ge=0, le=1)
    conscientiousness: float = Field(ge=0, le=1)
    extraversion: float = Field(ge=0, le=1)
    agreeableness: float = Field(ge=0, le=1)
    neuroticism: float = Field(ge=0, le=1)
    
    # Writing style traits
    formality_level: float = Field(ge=0, le=1)
    analytical_thinking: float = Field(ge=0, le=1)
    emotional_expressiveness: float = Field(ge=0, le=1)
    confidence_level: float = Field(ge=0, le=1)
    
    # Additional psychological insights
    dominant_cognitive_patterns: List[str]
    communication_preferences: List[str]
    thinking_style: str
    
class AuthorAnalysis(BaseModel):
    """Complete analysis of an author's writing style."""
    
    features: DocumentFeatures
    psychological_profile: PsychologicalProfile
    writing_style_summary: str
    distinguishing_characteristics: List[str]
    recommendations: List[str]

class DocumentAnalysisAgent:
    """
    An agent that analyzes documents to extract stylistic features and psychological traits.
    Uses the OpenAI Agents SDK for enhanced reasoning capabilities.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "gpt-4-turbo",
                 max_tokens_per_chunk: int = 8000,
                 max_files_per_analysis: int = 50):
        """
        Initialize the document analysis agent.
        
        Args:
            api_key: OpenAI API key. If None, it will be loaded from environment variables.
            model: The model to use for analysis.
            max_tokens_per_chunk: Maximum number of tokens to process in a single API call.
            max_files_per_analysis: Maximum number of files to include in a single analysis.
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.max_files_per_analysis = max_files_per_analysis
        
        # Initialize utility classes
        max_file_size = 10  # Default to 10MB max file size
        
        # Create a fresh directory traverser with minimal exclusions
        self.directory_traverser = DirectoryTraverser(
            max_file_size=max_file_size * 1024 * 1024,  # Convert MB to bytes
            excluded_dirs={'__pycache__'}  # Minimal exclusions
        )
        
        # Create document parser
        self.document_parser = DocumentParser(max_file_size=max_file_size * 1024 * 1024)
        
        # Log initialization
        print(f"DocumentAnalysisAgent: Initialized directory_traverser with max_file_size={max_file_size}MB")
        
        # Initialize the agent with tools
        self.agent = self._create_agent()
        
        # Track processed files
        self.processed_files: Set[str] = set()
        self.current_documents: List[Dict[str, Any]] = []
        
    def _create_agent(self) -> Agent:
        """Create an OpenAI Agent with the necessary tools for document analysis."""
        
        # Core analysis tools
        tools = [
            NamedAgentTool(
                name="extract_document_features",
                description="Extract detailed stylistic features from text",
                callable=self._extract_document_features,
            ),
            NamedAgentTool(
                name="analyze_psychological_traits",
                description="Analyze psychological traits based on writing style",
                callable=self._analyze_psychological_traits,
            ),
            NamedAgentTool(
                name="summarize_writing_style",
                description="Generate a concise summary of writing style",
                callable=self._summarize_writing_style,
            ),
        ]
        
        # Add file operation tools
        tools.extend(self.directory_traverser.get_agent_tools())
        tools.extend(self.document_parser.get_agent_tools())
        
        system_prompt = """You are an expert document analysis agent specializing in stylistic analysis and 
        psychological profiling based on writing patterns. Your task is to analyze documents and extract 
        insights about the author's writing style, vocabulary use, sentence structure, and psychological traits.
        
        Your analysis should be thorough, evidence-based, and nuanced. Avoid making definitive claims about 
        the author's psychology without sufficient evidence. Focus on patterns in the text that reveal 
        stylistic preferences and cognitive patterns.
        
        You have tools available to:
        1. Extract detailed stylistic features from text
        2. Analyze psychological traits based on writing style
        3. Summarize writing style concisely
        
        Process documents methodically and provide structured, objective analysis.
        """
        
        return Agent(
            system_prompt=system_prompt,
            tools=tools,
            model=self.model,
            api_key=self.api_key
        )
    
    def _extract_document_features(self, text: str) -> Dict[str, Any]:
        """
        Extract detailed stylistic features from the provided text.
        This implementation calls the API for enhanced reasoning.
        
        Args:
            text: The document text to analyze
            
        Returns:
            A dictionary of extracted features
        """
        system_prompt = """You are an expert linguistic analyst. Extract and quantify the following features from the provided text:
        
        1. Vocabulary statistics: vocabulary size, average word length, word frequencies, rare words
        2. Sentence structure: average length, variation in length, types of structures used
        3. Stylistic elements: idioms, metaphors, transition phrases
        4. Writing patterns: paragraph structures, punctuation usage, voice (active/passive)
        
        Provide quantitative metrics where possible, and qualitative analysis with specific examples from the text.
        Your response should be structured in JSON format.
        """
        
        try:
            # Check if API key is provided
            if not self.api_key:
                raise ValueError("OpenAI API key is required for document analysis. Please provide an API key.")
                
            # Use our wrapper to initialize OpenAI client safely
            client = OpenAIClientWrapper(api_key=self.api_key, model=self.model)
            
            # Use the client wrapper to create a chat completion
            response = client.create_chat_completion(
                system_prompt=system_prompt,
                user_prompt=text,
                response_format={"type": "json_object"}
            )
            
            features = json.loads(response.choices[0].message.content)
            return features
        except ValueError as e:
            # Re-raise ValueError for API key issues
            raise e
        except Exception as e:
            # Log and re-raise other exceptions with more context
            error_msg = f"Error extracting document features: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def _analyze_psychological_traits(self, text: str) -> Dict[str, Any]:
        """
        Analyze psychological traits based on writing style in the provided text.
        
        Args:
            text: The document text to analyze
            
        Returns:
            A dictionary of psychological traits
        """
        system_prompt = """You are an expert in psycholinguistics and writing analysis. Analyze the provided text to infer 
        psychological traits and cognitive patterns of the author. Include:
        
        1. Estimated Big Five personality traits (openness, conscientiousness, extraversion, agreeableness, neuroticism)
        2. Writing style traits (formality, analytical thinking, emotional expressiveness, confidence)
        3. Dominant cognitive patterns and communication preferences
        
        Express metrics on a scale from 0.0 to 1.0. Be careful not to overstate confidence in these inferences.
        Provide specific textual evidence for each trait identified.
        Your response should be structured in JSON format.
        """
        
        try:
            # Check if API key is provided
            if not self.api_key:
                raise ValueError("OpenAI API key is required for document analysis. Please provide an API key.")
                
            # Use our wrapper to initialize OpenAI client safely
            client = OpenAIClientWrapper(api_key=self.api_key, model=self.model)
            
            # Use the client wrapper to create a chat completion
            response = client.create_chat_completion(
                system_prompt=system_prompt,
                user_prompt=text,
                response_format={"type": "json_object"}
            )
            
            psychological_traits = json.loads(response.choices[0].message.content)
            return psychological_traits
        except ValueError as e:
            # Re-raise ValueError for API key issues
            raise e
        except Exception as e:
            # Log and re-raise other exceptions with more context
            error_msg = f"Error analyzing psychological traits: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def _summarize_writing_style(self, features: Dict[str, Any], psychological_traits: Dict[str, Any]) -> str:
        """
        Generate a concise summary of writing style based on features and psychological traits.
        
        Args:
            features: Extracted document features
            psychological_traits: Analyzed psychological traits
            
        Returns:
            A concise summary of the writing style
        """
        system_prompt = """You are an expert writing analyst. Based on the provided document features and 
        psychological traits, create a concise, insightful summary of the author's writing style.
        
        Focus on the most distinctive aspects of their style, unusual patterns, and what makes 
        their writing unique. Avoid generic descriptions that could apply to many writers.
        
        Your summary should be 3-5 paragraphs and should synthesize both the technical aspects 
        of the writing (vocabulary, sentence structure, etc.) and the psychological dimensions.
        """
        
        try:
            # Prepare the input for the API
            input_data = json.dumps({
                "document_features": features,
                "psychological_traits": psychological_traits
            })
            
            # Check if API key is provided
            if not self.api_key:
                raise ValueError("OpenAI API key is required for document analysis. Please provide an API key.")
                
            # Use our wrapper to initialize OpenAI client safely
            client = OpenAIClientWrapper(api_key=self.api_key, model=self.model)
            
            # Use the client wrapper to create a chat completion
            response = client.create_chat_completion(
                system_prompt=system_prompt,
                user_prompt=input_data
            )
            
            return response.choices[0].message.content
        except ValueError as e:
            # Re-raise ValueError for API key issues
            raise e
        except Exception as e:
            # Log and re-raise other exceptions with more context
            error_msg = f"Error summarizing writing style: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def read_documents(self, folder_path: str, file_extensions: List[str] = None) -> List[Dict[str, Any]]:
        """
        Recursively read all document files in the given folder with specified extensions.
        Uses the DirectoryTraverser and DocumentParser for enhanced file handling.
        
        Args:
            folder_path: Path to the folder containing documents
            file_extensions: List of file extensions to include (if None, uses supported formats from DocumentParser)
            
        Returns:
            List of document objects with metadata and content
        """
        documents = []
        file_count = 0
        
        # If no extensions provided, use the supported formats from DocumentParser
        if file_extensions is None:
            file_extensions = self.document_parser.get_supported_formats()
        
        # Convert extensions to match the format used by find_files
        file_patterns = [f"*{ext}" if ext.startswith('.') else f"*.{ext}" for ext in file_extensions]
        
        # Find all matching files using DirectoryTraverser
        for pattern in file_patterns:
            if file_count >= self.max_files_per_analysis:
                break
                
            matching_files = self.directory_traverser.find_files(
                directory_path=folder_path,
                pattern=pattern,
                recursive=True,
                include_binary=True  # DocumentParser will handle binary files
            )
            
            for file_metadata in matching_files:
                # Check if we've reached the maximum number of files
                if file_count >= self.max_files_per_analysis:
                    print(f"Warning: Reached maximum of {self.max_files_per_analysis} files. Some files will be skipped.")
                    break
                
                # Skip if we've already processed this file
                if file_metadata.path in self.processed_files:
                    continue
                
                try:
                    # Use DocumentParser to parse the file
                    doc_content = self.document_parser.parse_document(file_metadata.path)
                    
                    # Get relative path for consistent references
                    try:
                        relative_path = os.path.relpath(file_metadata.path, folder_path)
                    except ValueError:
                        # Fallback if paths are on different drives
                        relative_path = file_metadata.path
                    
                    documents.append({
                        'path': relative_path,
                        'name': file_metadata.name,
                        'content': doc_content.text_content,
                        'size': file_metadata.size,
                        'modified': file_metadata.modified,
                        'created': file_metadata.created,
                        'metadata': doc_content.metadata
                    })
                    
                    # Mark as processed
                    self.processed_files.add(file_metadata.path)
                    file_count += 1
                except Exception as e:
                    print(f"Error reading file {file_metadata.path}: {e}")
        
        # Store the current documents for analysis
        self.current_documents = documents
        
        return documents
    
    def _prepare_document_batches(self, max_batch_tokens: int = 8000) -> List[str]:
        """
        Prepare batches of document content for processing, respecting token limits.
        
        Args:
            max_batch_tokens: Maximum number of tokens in a batch
            
        Returns:
            List of text batches ready for processing
        """
        batches = []
        current_batch = ""
        current_token_estimate = 0
        
        # Simple token estimation: ~4 characters per token on average
        chars_per_token = 4
        
        for doc in self.current_documents:
            content = doc['content']
            content_token_estimate = len(content) // chars_per_token
            
            # If adding this document would exceed the limit, start a new batch
            if current_token_estimate + content_token_estimate > max_batch_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = f"--- Document: {doc['name']} ---\n\n{content}\n\n"
                current_token_estimate = content_token_estimate
            else:
                current_batch += f"--- Document: {doc['name']} ---\n\n{content}\n\n"
                current_token_estimate += content_token_estimate
        
        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)
            
        return batches
    
    def analyze_documents(self, folder_path: str, file_extensions: List[str] = ['.txt', '.md']) -> AuthorAnalysis:
        """
        Analyze all documents in the specified folder to generate a complete author analysis.
        
        Args:
            folder_path: Path to the folder containing documents
            file_extensions: List of file extensions to include
            
        Returns:
            AuthorAnalysis object containing the complete analysis
        """
        # Read all documents in the folder
        documents = self.read_documents(folder_path, file_extensions)
        
        if not documents:
            # Try with more verbose debugging
            print(f"Trying to find documents in {folder_path} with extensions {file_extensions}")
            for ext in file_extensions:
                pattern = f"*{ext}" if ext.startswith('.') else f"*.{ext}"
                files = self.directory_traverser.find_files(folder_path, pattern, recursive=True)
                if files:
                    print(f"Found {len(files)} files matching {pattern}:")
                    for f in files:
                        print(f"  - {f.path}")
                else:
                    print(f"No files found matching {pattern}")
            
            # If we really have no documents, raise the error
            raise ValueError(f"No documents found in {folder_path} with extensions {file_extensions}")
            
        print(f"Processing {len(documents)} documents...")
        
        # Prepare document batches
        text_batches = self._prepare_document_batches(self.max_tokens_per_chunk)
        
        # Process each batch to extract features
        all_features = []
        all_psychological_traits = []
        
        for i, batch in enumerate(text_batches):
            print(f"Processing batch {i+1} of {len(text_batches)}...")
            
            # Extract features from the batch
            features = self._extract_document_features(batch)
            all_features.append(features)
            
            # Analyze psychological traits
            psychological_traits = self._analyze_psychological_traits(batch)
            all_psychological_traits.append(psychological_traits)
            
        # Merge results from all batches
        merged_features = self._merge_features(all_features)
        merged_psychological_traits = self._merge_psychological_traits(all_psychological_traits)
        
        # Generate summary
        writing_style_summary = self._summarize_writing_style(merged_features, merged_psychological_traits)
        
        # Use agent-based reasoning to identify distinguishing characteristics
        distinguishing_characteristics = self._identify_distinguishing_characteristics(
            merged_features, merged_psychological_traits
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            merged_features, merged_psychological_traits
        )
        
        # Create the complete analysis
        return AuthorAnalysis(
            features=DocumentFeatures(**merged_features),
            psychological_profile=PsychologicalProfile(**merged_psychological_traits),
            writing_style_summary=writing_style_summary,
            distinguishing_characteristics=distinguishing_characteristics,
            recommendations=recommendations
        )
    
    def _merge_features(self, feature_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge document features from multiple batches.
        
        Args:
            feature_list: List of feature dictionaries from different batches
            
        Returns:
            Merged features dictionary
        """
        if not feature_list:
            return {}
            
        # For demonstration, this is a simplified merging strategy
        # In a real implementation, this would be more sophisticated
        
        merged = defaultdict(int)
        list_fields = set()
        dict_fields = set()
        
        # First pass: identify field types
        for features in feature_list:
            for key, value in features.items():
                if isinstance(value, list):
                    list_fields.add(key)
                elif isinstance(value, dict):
                    dict_fields.add(key)
                    
        # Second pass: merge values based on type
        for features in feature_list:
            for key, value in features.items():
                if key in list_fields:
                    if key not in merged:
                        merged[key] = []
                    merged[key].extend(value)
                elif key in dict_fields:
                    if key not in merged:
                        merged[key] = defaultdict(int)
                    for k, v in value.items():
                        if isinstance(v, int):
                            merged[key][k] += v
                        else:
                            merged[key][k] = v
                elif isinstance(value, (int, float)):
                    merged[key] += value
                else:
                    # For other types, just use the last value
                    merged[key] = value
        
        # Average out numerical values that should be averaged
        avg_fields = {'average_word_length', 'average_sentence_length', 
                     'sentence_length_variation', 'passive_voice_frequency',
                     'active_voice_frequency'}
        
        for field in avg_fields:
            if field in merged:
                merged[field] = merged[field] / len(feature_list)
                
        # Remove duplicates from lists
        for field in list_fields:
            if field in merged:
                merged[field] = list(set(merged[field]))
        
        return dict(merged)
    
    def _merge_psychological_traits(self, trait_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge psychological traits from multiple batches.
        
        Args:
            trait_list: List of psychological trait dictionaries
            
        Returns:
            Merged psychological traits dictionary
        """
        if not trait_list:
            return {}
            
        merged = {}
        numeric_fields = {
            'openness', 'conscientiousness', 'extraversion', 'agreeableness', 
            'neuroticism', 'formality_level', 'analytical_thinking', 
            'emotional_expressiveness', 'confidence_level'
        }
        list_fields = {'dominant_cognitive_patterns', 'communication_preferences'}
        
        # Merge numeric fields (average them)
        for field in numeric_fields:
            values = [traits.get(field, 0) for traits in trait_list if field in traits]
            if values:
                merged[field] = sum(values) / len(values)
            else:
                merged[field] = 0.5  # Default middle value
        
        # Merge lists (combine and deduplicate)
        for field in list_fields:
            merged[field] = []
            for traits in trait_list:
                if field in traits and traits[field]:
                    merged[field].extend(traits[field])
            merged[field] = list(set(merged[field]))
        
        # For thinking_style, use the most frequent one
        thinking_styles = [traits.get('thinking_style') for traits in trait_list 
                          if 'thinking_style' in traits]
        if thinking_styles:
            merged['thinking_style'] = Counter(thinking_styles).most_common(1)[0][0]
        else:
            merged['thinking_style'] = "analytical"  # Default
            
        return merged
    
    def _identify_distinguishing_characteristics(self, 
                                              features: Dict[str, Any], 
                                              psychological_traits: Dict[str, Any]) -> List[str]:
        """
        Identify the most distinguishing characteristics of the author's style.
        
        Args:
            features: Merged document features
            psychological_traits: Merged psychological traits
            
        Returns:
            List of distinguishing characteristics
        """
        system_prompt = """You are an expert writing analyst. Based on the provided document features and 
        psychological traits, identify the 5-7 most distinguishing characteristics of this author's writing.
        
        Focus on what makes this writing unique or unusual compared to typical writing. Look for outliers 
        in the data, unique combinations of traits, or distinctive patterns.
        
        Format your response as a JSON array of strings, each describing one distinguishing characteristic.
        """
        
        try:
            # Prepare the input for the API
            input_data = json.dumps({
                "document_features": features,
                "psychological_traits": psychological_traits
            })
            
            # Check if API key is provided
            if not self.api_key:
                raise ValueError("OpenAI API key is required for document analysis. Please provide an API key.")
                
            # Use our wrapper to initialize OpenAI client safely
            client = OpenAIClientWrapper(api_key=self.api_key, model=self.model)
            
            # Use the client wrapper to create a chat completion
            response = client.create_chat_completion(
                system_prompt=system_prompt,
                user_prompt=input_data,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # The result should be a JSON object with a list
            if isinstance(result, dict) and "characteristics" in result:
                return result["characteristics"]
            elif isinstance(result, list):
                return result
            else:
                # Handle unexpected format
                return []
        except ValueError as e:
            # Re-raise ValueError for API key issues
            raise e
        except Exception as e:
            # Log and re-raise other exceptions with more context
            error_msg = f"Error identifying distinguishing characteristics: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def _generate_recommendations(self, 
                               features: Dict[str, Any], 
                               psychological_traits: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for writing improvement or stylistic development.
        
        Args:
            features: Merged document features
            psychological_traits: Merged psychological traits
            
        Returns:
            List of recommendations
        """
        system_prompt = """You are an expert writing coach. Based on the provided document features and 
        psychological traits, provide 3-5 specific recommendations for the author to:
        
        1. Enhance their strengths
        2. Address potential weaknesses
        3. Develop their unique voice further
        
        Make recommendations specific and actionable, with concrete examples where possible.
        Format your response as a JSON array of strings, each containing one recommendation.
        """
        
        try:
            # Prepare the input for the API
            input_data = json.dumps({
                "document_features": features,
                "psychological_traits": psychological_traits
            })
            
            # Check if API key is provided
            if not self.api_key:
                raise ValueError("OpenAI API key is required for document analysis. Please provide an API key.")
                
            # Use our wrapper to initialize OpenAI client safely
            client = OpenAIClientWrapper(api_key=self.api_key, model=self.model)
            
            # Use the client wrapper to create a chat completion
            response = client.create_chat_completion(
                system_prompt=system_prompt,
                user_prompt=input_data,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # The result should be a JSON object with a list
            if isinstance(result, dict) and "recommendations" in result:
                return result["recommendations"]
            elif isinstance(result, list):
                return result
            else:
                # Handle unexpected format
                return []
        except ValueError as e:
            # Re-raise ValueError for API key issues
            raise e
        except Exception as e:
            # Log and re-raise other exceptions with more context
            error_msg = f"Error generating recommendations: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
            
    def save_analysis(self, analysis: AuthorAnalysis, output_path: str) -> str:
        """
        Save the analysis results to a file.
        
        Args:
            analysis: The complete author analysis
            output_path: Path to save the analysis to
            
        Returns:
            The path to the saved file
        """
        # Ensure the directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Convert the analysis to a dictionary
        analysis_dict = analysis.model_dump()
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, indent=2)
            
        return output_path
        
    def generate_analysis_report(self, analysis: AuthorAnalysis, output_path: str) -> str:
        """
        Generate a human-readable report from the analysis.
        
        Args:
            analysis: The complete author analysis
            output_path: Path to save the report to
            
        Returns:
            The path to the saved report
        """
        # Create a markdown report
        report = f"""# Author Writing Style Analysis

## Summary

{analysis.writing_style_summary}

## Distinguishing Characteristics

"""
        
        for characteristic in analysis.distinguishing_characteristics:
            report += f"- {characteristic}\n"
            
        report += "\n## Vocabulary and Language\n\n"
        report += f"- Vocabulary size: {analysis.features.vocabulary_size}\n"
        report += f"- Average word length: {analysis.features.average_word_length:.2f} characters\n"
        
        # Add top words section
        report += "\n### Most Frequently Used Words\n\n"
        sorted_words = sorted(
            analysis.features.word_frequencies.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20]  # Top 20 words
        
        for word, frequency in sorted_words:
            report += f"- {word}: {frequency}\n"
            
        report += "\n## Sentence Structure\n\n"
        report += f"- Average sentence length: {analysis.features.average_sentence_length:.2f} words\n"
        report += f"- Sentence length variation: {analysis.features.sentence_length_variation:.2f}\n"
        
        # Add sentence structures
        report += "\n### Sentence Structure Distribution\n\n"
        for structure, count in analysis.features.sentence_structures.items():
            report += f"- {structure}: {count}\n"
            
        report += "\n## Psychological Profile\n\n"
        
        # Big Five personality traits
        report += "### Big Five Personality Traits\n\n"
        report += f"- Openness: {analysis.psychological_profile.openness * 100:.0f}%\n"
        report += f"- Conscientiousness: {analysis.psychological_profile.conscientiousness * 100:.0f}%\n"
        report += f"- Extraversion: {analysis.psychological_profile.extraversion * 100:.0f}%\n"
        report += f"- Agreeableness: {analysis.psychological_profile.agreeableness * 100:.0f}%\n"
        report += f"- Neuroticism: {analysis.psychological_profile.neuroticism * 100:.0f}%\n"
        
        # Writing style traits
        report += "\n### Writing Style Traits\n\n"
        report += f"- Formality level: {analysis.psychological_profile.formality_level * 100:.0f}%\n"
        report += f"- Analytical thinking: {analysis.psychological_profile.analytical_thinking * 100:.0f}%\n"
        report += f"- Emotional expressiveness: {analysis.psychological_profile.emotional_expressiveness * 100:.0f}%\n"
        report += f"- Confidence level: {analysis.psychological_profile.confidence_level * 100:.0f}%\n"
        
        # Thinking style
        report += f"\n### Thinking Style\n\n{analysis.psychological_profile.thinking_style}\n\n"
        
        # Cognitive patterns and communication preferences
        report += "### Dominant Cognitive Patterns\n\n"
        for pattern in analysis.psychological_profile.dominant_cognitive_patterns:
            report += f"- {pattern}\n"
            
        report += "\n### Communication Preferences\n\n"
        for preference in analysis.psychological_profile.communication_preferences:
            report += f"- {preference}\n"
            
        # Recommendations
        report += "\n## Recommendations\n\n"
        for recommendation in analysis.recommendations:
            report += f"- {recommendation}\n"
            
        # Meta statistics
        report += f"\n## Analysis Metadata\n\n"
        report += f"- Documents analyzed: {analysis.features.document_count}\n"
        report += f"- Total word count: {analysis.features.total_word_count}\n"
        report += f"- Total sentences: {analysis.features.total_sentence_count}\n"
        
        # Save the report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        return output_path
