"""
Persona serialization tools for Simulacra.

This module provides functionality to save generated personas to disk in structured formats,
with integration with the OpenAI Agents SDK.
"""

import os
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, BinaryIO
from ..openai_agents import AgentTool, NamedAgentTool
from pydantic import BaseModel, Field

from ..persona_generator.persona import Persona

class SerializationResult(BaseModel):
    """Result of a persona serialization operation."""
    path: str
    format: str
    size: int
    success: bool
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PersonaSerializer:
    """
    A tool for saving generated personas to disk in structured formats.
    Integrates with the OpenAI Agents SDK.
    """
    
    def __init__(self, 
                 default_output_dir: str = "./output/personas",
                 auto_create_dirs: bool = True):
        """
        Initialize the persona serializer.
        
        Args:
            default_output_dir: Default directory to save personas to
            auto_create_dirs: Whether to automatically create directories
        """
        self.default_output_dir = default_output_dir
        self.auto_create_dirs = auto_create_dirs
        self._supported_formats = ['json', 'yaml', 'markdown', 'md', 'txt']
        
    def get_agent_tools(self) -> List[NamedAgentTool]:
        """Get the agent tools for persona serialization."""
        
        return [
            NamedAgentTool(
                name="save_persona",
                description="Save a persona to disk in the specified format",
                callable=self.save_persona,
            ),
            NamedAgentTool(
                name="load_persona",
                description="Load a persona from disk",
                callable=self.load_persona,
            ),
            NamedAgentTool(
                name="get_supported_formats",
                description="Get a list of supported serialization formats",
                callable=self.get_supported_formats,
            ),
        ]
    
    def save_persona(self, 
                    persona: Persona, 
                    output_path: Optional[str] = None, 
                    format: str = 'json',
                    overwrite: bool = False) -> SerializationResult:
        """
        Save a persona to disk in the specified format.
        
        Args:
            persona: The persona to save
            output_path: Path to save the persona to (if None, uses default with persona name)
            format: Format to save in ('json', 'yaml', 'markdown', 'md', 'txt')
            overwrite: Whether to overwrite existing files
            
        Returns:
            SerializationResult object with the result of the operation
        """
        # Validate format
        format = format.lower()
        if format not in self._supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported formats: {', '.join(self._supported_formats)}")
        
        # Set default output path if not provided
        if output_path is None:
            # Generate a filename based on persona name and format
            safe_name = self._get_safe_filename(persona.name)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{safe_name}_{timestamp}.{format}"
            output_path = os.path.join(self.default_output_dir, filename)
        
        # Check if file exists and handle overwrite
        output_file = Path(output_path)
        if output_file.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {output_path}. Use overwrite=True to overwrite.")
        
        # Ensure directory exists
        if self.auto_create_dirs:
            output_dir = output_file.parent
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Serialize based on format
        try:
            if format == 'json':
                content = self._serialize_to_json(persona)
            elif format == 'yaml':
                content = self._serialize_to_yaml(persona)
            elif format in ['markdown', 'md', 'txt']:
                content = self._serialize_to_markdown(persona)
            else:
                # Should not happen due to validation above
                raise ValueError(f"Unsupported format: {format}")
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Get file metadata
            file_stat = os.stat(output_path)
            
            return SerializationResult(
                path=output_path,
                format=format,
                size=file_stat.st_size,
                success=True,
                timestamp=datetime.datetime.now().isoformat(),
                metadata={
                    'persona_name': persona.name,
                    'persona_id': persona.id,
                    'traits_count': len(persona.traits),
                    'knowledge_areas_count': len(persona.knowledge_areas)
                }
            )
        except Exception as e:
            # Handle serialization errors
            error_result = SerializationResult(
                path=output_path,
                format=format,
                size=0,
                success=False,
                timestamp=datetime.datetime.now().isoformat(),
                metadata={'error': str(e)}
            )
            return error_result
    
    def load_persona(self, file_path: str) -> Persona:
        """
        Load a persona from disk.
        
        Args:
            file_path: Path to the persona file
            
        Returns:
            The loaded Persona object
        """
        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file.suffix.lower()
        
        try:
            # Load based on file extension
            if extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return Persona.from_dict(data)
            elif extension == '.yaml' or extension == '.yml':
                return self._load_from_yaml(file_path)
            elif extension in ['.md', '.markdown', '.txt']:
                return self._load_from_markdown(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {extension}")
        except Exception as e:
            raise ValueError(f"Error loading persona from {file_path}: {e}")
    
    def get_supported_formats(self) -> List[str]:
        """
        Get a list of supported serialization formats.
        
        Returns:
            List of supported formats
        """
        return self._supported_formats
    
    def _serialize_to_json(self, persona: Persona) -> str:
        """Serialize a persona to JSON format."""
        data = persona.to_dict()
        return json.dumps(data, indent=2)
    
    def _serialize_to_yaml(self, persona: Persona) -> str:
        """Serialize a persona to YAML format."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML serialization. Install it with 'pip install PyYAML'.")
        
        data = persona.to_dict()
        return yaml.dump(data, sort_keys=False, default_flow_style=False)
    
    def _serialize_to_markdown(self, persona: Persona) -> str:
        """Serialize a persona to Markdown format."""
        md = f"# {persona.name}\n\n"
        
        # Traits section
        md += "## Traits\n\n"
        for trait in persona.traits:
            md += f"- {trait}\n"
        
        # Background section
        md += "\n## Background\n\n"
        md += f"{persona.background}\n\n"
        
        # Communication style
        md += "## Communication Style\n\n"
        md += f"{persona.communication_style}\n\n"
        
        # Knowledge areas
        if persona.knowledge_areas:
            md += "## Areas of Expertise\n\n"
            for area in persona.knowledge_areas:
                md += f"- {area}\n"
            md += "\n"
        
        # Additional details
        if persona.additional_details:
            md += "## Additional Details\n\n"
            for key, value in persona.additional_details.items():
                if isinstance(value, (str, int, float, bool)):
                    md += f"### {key.replace('_', ' ').title()}\n\n"
                    md += f"{value}\n\n"
                elif isinstance(value, list):
                    md += f"### {key.replace('_', ' ').title()}\n\n"
                    for item in value:
                        md += f"- {item}\n"
                    md += "\n"
                elif isinstance(value, dict):
                    md += f"### {key.replace('_', ' ').title()}\n\n"
                    for k, v in value.items():
                        md += f"- **{k}**: {v}\n"
                    md += "\n"
        
        # Metadata
        md += "---\n"
        md += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        if persona.id:
            md += f"ID: {persona.id}\n"
        
        return md
    
    def _load_from_yaml(self, file_path: str) -> Persona:
        """Load a persona from YAML format."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML deserialization. Install it with 'pip install PyYAML'.")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return Persona.from_dict(data)
    
    def _load_from_markdown(self, file_path: str) -> Persona:
        """Load a persona from Markdown format."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Initialize persona fields
        data = {
            "name": "",
            "traits": [],
            "background": "",
            "communication_style": "",
            "knowledge_areas": [],
            "additional_details": {}
        }
        
        # Extract name from heading (# Name)
        name_match = content.split('\n')[0]
        if name_match.startswith('# '):
            data["name"] = name_match[2:].strip()
        
        # Parse sections
        sections = {}
        current_section = None
        section_content = []
        
        # Split content by lines for processing
        lines = content.split('\n')
        i = 1  # Skip the title which we already processed
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this is a new section heading
            if line.startswith('## '):
                # Save previous section if it exists
                if current_section:
                    sections[current_section] = '\n'.join(section_content).strip()
                
                # Start new section
                current_section = line[3:].strip()
                section_content = []
            # Check if this is a subsection
            elif line.startswith('### ') and current_section == "Additional Details":
                # Process subsection
                subsection = line[4:].strip()
                subsection_content = []
                
                # Move to next line
                i += 1
                
                # Collect subsection content until next subsection or section
                while i < len(lines) and not lines[i].startswith('## ') and not lines[i].startswith('### ') and not lines[i].startswith('---'):
                    if lines[i].strip():  # Skip empty lines
                        subsection_content.append(lines[i])
                    i += 1
                
                # Store subsection content
                subsection_text = '\n'.join(subsection_content).strip()
                # Convert to snake_case for key
                subsection_key = subsection.lower().replace(' ', '_')
                
                # Handle lists in subsections
                if all(line.strip().startswith('- ') for line in subsection_content if line.strip()):
                    # It's a list
                    data["additional_details"][subsection_key] = [
                        item[2:].strip() for item in subsection_content if item.strip()
                    ]
                else:
                    # It's a text value
                    data["additional_details"][subsection_key] = subsection_text
                
                # Continue from current position in next iteration
                i -= 1
            # Check if this is the metadata section
            elif line.startswith('---'):
                # End the previous section
                if current_section:
                    sections[current_section] = '\n'.join(section_content).strip()
                
                # Process metadata
                i += 1
                while i < len(lines):
                    if lines[i].startswith('ID: '):
                        data["id"] = lines[i][4:].strip()
                    i += 1
                
                break  # End processing
            else:
                section_content.append(line)
            
            i += 1
        
        # Save the last section if we didn't encounter metadata
        if current_section and current_section not in sections:
            sections[current_section] = '\n'.join(section_content).strip()
        
        # Process extracted sections
        if "Traits" in sections:
            traits_content = sections["Traits"]
            # Extract bullet points
            data["traits"] = [
                line[2:].strip() for line in traits_content.split('\n') 
                if line.strip().startswith('- ')
            ]
        
        if "Background" in sections:
            data["background"] = sections["Background"]
        
        if "Communication Style" in sections:
            data["communication_style"] = sections["Communication Style"]
        
        if "Areas of Expertise" in sections:
            expertise_content = sections["Areas of Expertise"]
            # Extract bullet points
            data["knowledge_areas"] = [
                line[2:].strip() for line in expertise_content.split('\n') 
                if line.strip().startswith('- ')
            ]
        
        # Create and return the persona
        return Persona.from_dict(data)

    def _get_safe_filename(self, name: str) -> str:
        """Convert a string to a safe filename."""
        # Replace spaces with underscores and remove invalid characters
        safe_name = name.replace(' ', '_')
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '_-')
        return safe_name
