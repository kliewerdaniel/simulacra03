import os
from typing import Dict, Any, Optional, List
import json
import requests


class ResponsesAPIClient:
    """Client for interacting with the OpenAI Responses API using direct HTTP requests."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        """
        Initialize the API client.
        
        Args:
            api_key: OpenAI API key. If None, it will be loaded from environment variables.
            model: The model to use for generating responses.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        self.model = model
        self.api_base = "https://api.openai.com/v1"
        
    def generate_response(self, 
                        system_message: str, 
                        user_message: str,
                        temperature: float = 0.7,
                        max_tokens: int = 500,
                        **kwargs) -> str:
        """
        Generate a response using the OpenAI API.
        
        Args:
            system_message: The system message that sets the context
            user_message: The user message to respond to
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to the OpenAI API
            
        Returns:
            The generated response as a string
        """
        try:
            # Create the messages array
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                payload[key] = value
            
            # Define headers with API key
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make the API call directly with requests
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload
            )
            
            # Raise an exception if the request failed
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Extract the content from the response
            return response_data["choices"][0]["message"]["content"]
            
        except Exception as e:
            import traceback
            print(f"Error in generate_response: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to generate response: {e}")
    
    def generate_agent_response(self, agent_id: str, messages: list, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using an agent through the API.
        
        Args:
            agent_id: The ID of the agent to use
            messages: The list of messages in the conversation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The response from the agent
        """
        try:
            # Prepare the request payload
            payload = {
                "agent_id": agent_id,
                "messages": messages
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                payload[key] = value
            
            # Define headers with API key
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make the API call directly with requests
            response = requests.post(
                f"{self.api_base}/beta/agents/messages",
                headers=headers,
                json=payload
            )
            
            # Raise an exception if the request failed
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Extract the content from the response
            return response_data["choices"][0]["message"]["content"]
            
        except Exception as e:
            import traceback
            print(f"Error in generate_agent_response: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to generate agent response: {e}")
