from typing import List, Dict, Any, Optional, Union
from uuid import uuid4

# Use local stub implementation instead of the package
from ..openai_agents import AgentTool, NamedAgentTool, AgentAction, Agent

from .persona import Persona


class PersonaGenerator:
    """A class for generating AI personas using the OpenAI Agents SDK."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the PersonaGenerator.
        
        Args:
            api_key: OpenAI API key. If None, it will be loaded from environment variables.
        """
        self.api_key = api_key
        self.personas: Dict[str, Persona] = {}
        
    def create_persona(self, 
                      name: str, 
                      traits: List[str], 
                      background: str, 
                      communication_style: str,
                      knowledge_areas: Optional[List[str]] = None,
                      additional_details: Optional[Dict[str, Any]] = None) -> Persona:
        """
        Create a new AI persona with the specified traits and characteristics.
        
        Args:
            name: The name of the persona
            traits: List of personality traits
            background: Background story or context
            communication_style: Description of how the persona communicates
            knowledge_areas: Optional list of areas of expertise
            additional_details: Optional dictionary of additional details
            
        Returns:
            A new Persona instance
        """
        persona_id = str(uuid4())
        
        # Create the persona
        persona = Persona(
            id=persona_id,
            name=name,
            traits=traits,
            background=background,
            communication_style=communication_style,
            knowledge_areas=knowledge_areas or [],
            additional_details=additional_details or {}
        )
        
        # Store the persona
        self.personas[persona_id] = persona
        
        return persona
    
    def get_persona(self, persona_id: str) -> Optional[Persona]:
        """
        Retrieve a persona by ID.
        
        Args:
            persona_id: The ID of the persona to retrieve
            
        Returns:
            The Persona if found, None otherwise
        """
        return self.personas.get(persona_id)
    
    def update_persona(self, persona_id: str, **kwargs) -> Optional[Persona]:
        """
        Update an existing persona.
        
        Args:
            persona_id: The ID of the persona to update
            **kwargs: The fields to update
            
        Returns:
            The updated Persona if found, None otherwise
        """
        persona = self.get_persona(persona_id)
        if not persona:
            return None
        
        # Update the persona fields
        for key, value in kwargs.items():
            if hasattr(persona, key):
                setattr(persona, key, value)
        
        return persona
    
    def delete_persona(self, persona_id: str) -> bool:
        """
        Delete a persona by ID.
        
        Args:
            persona_id: The ID of the persona to delete
            
        Returns:
            True if the persona was deleted, False otherwise
        """
        if persona_id in self.personas:
            del self.personas[persona_id]
            return True
        return False
    
    def list_personas(self) -> List[Persona]:
        """
        List all personas.
        
        Returns:
            A list of all personas
        """
        return list(self.personas.values())
    
    def create_agent_from_persona(self, persona_id: str, 
                                 tools: Optional[List[Union[AgentTool, NamedAgentTool]]] = None) -> Optional[Agent]:
        """
        Create an OpenAI Agent from a persona.
        
        Args:
            persona_id: The ID of the persona to use
            tools: Optional list of tools for the agent
            
        Returns:
            An OpenAI Agent if the persona exists, None otherwise
        """
        persona = self.get_persona(persona_id)
        if not persona:
            return None
        
        # Generate the system message for the agent
        system_message = persona._generate_system_message()
        
        # Create the agent
        agent = Agent(
            system_prompt=system_message,
            tools=tools or [],
            model="gpt-4-turbo",
            api_key=self.api_key
        )
        
        return agent
    
    def simulate_conversation(self, persona1_id: str, persona2_id: str, 
                             initial_message: str, num_turns: int = 3) -> List[Dict[str, Any]]:
        """
        Simulate a conversation between two personas.
        
        Args:
            persona1_id: The ID of the first persona
            persona2_id: The ID of the second persona
            initial_message: The message to start the conversation
            num_turns: The number of turns in the conversation
            
        Returns:
            A list of message dictionaries containing the conversation
        """
        persona1 = self.get_persona(persona1_id)
        persona2 = self.get_persona(persona2_id)
        
        if not persona1 or not persona2:
            raise ValueError("Both personas must exist")
        
        conversation = [
            {"role": "system", "content": "The following is a conversation between two personas."},
            {"role": "user", "content": f"{persona1.name}: {initial_message}"}
        ]
        
        current_persona = persona2
        other_persona = persona1
        
        for _ in range(num_turns):
            # Get the conversation history for context
            conversation_text = "\n".join([
                f"{msg['role'].title()}: {msg['content']}" for msg in conversation
            ])
            
            # Generate a response from the current persona
            prompt = f"You are {current_persona.name}. The following is your conversation with {other_persona.name}. Respond in character.\n\n{conversation_text}"
            response = current_persona.generate_response(prompt)
            
            # Add the response to the conversation
            conversation.append({
                "role": "assistant" if _ % 2 == 0 else "user",
                "content": f"{current_persona.name}: {response}"
            })
            
            # Switch personas for the next turn
            current_persona, other_persona = other_persona, current_persona
        
        return conversation
