from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Persona(BaseModel):
    """A class representing an AI persona with specific traits and characteristics."""
    
    id: Optional[str] = None
    name: str
    traits: List[str]
    background: str
    communication_style: str
    knowledge_areas: List[str] = Field(default_factory=list)
    additional_details: Dict[str, Any] = Field(default_factory=dict)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from this persona based on the input prompt.
        
        Args:
            prompt: The input text to respond to
            **kwargs: Additional parameters to pass to the OpenAI API
            
        Returns:
            A string containing the generated response
        """
        from .api_client import ResponsesAPIClient
        
        # Initialize client with only the API key to avoid 'proxies' parameter issue
        client = ResponsesAPIClient()
        
        # Construct a system message that describes the persona
        system_message = self._generate_system_message()
        
        # Generate the response using the OpenAI API
        response = client.generate_response(
            system_message=system_message,
            user_message=prompt,
            **kwargs
        )
        
        return response
    
    def _generate_system_message(self) -> str:
        """Generate a system message that describes this persona."""
        
        message = f"""You are {self.name}, with the following traits: {', '.join(self.traits)}.
Background: {self.background}
Communication style: {self.communication_style}
"""
        
        if self.knowledge_areas:
            message += f"Areas of expertise: {', '.join(self.knowledge_areas)}\n"
            
        if self.additional_details:
            for key, value in self.additional_details.items():
                message += f"{key}: {value}\n"
                
        message += "\nRespond to the user in a way that reflects these traits and background."
        
        return message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the persona to a dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Persona":
        """Create a persona from a dictionary."""
        return cls(**data)
