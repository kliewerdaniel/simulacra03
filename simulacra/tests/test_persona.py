import unittest
from unittest.mock import patch, MagicMock

from src.persona_generator import Persona


class TestPersona(unittest.TestCase):
    """Test cases for the Persona class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.persona = Persona(
            id="test-id",
            name="Test Persona",
            traits=["analytical", "calm", "precise"],
            background="AI researcher with expertise in natural language processing",
            communication_style="clear and concise, with a focus on accuracy"
        )
    
    def test_initialization(self):
        """Test that a persona can be properly initialized."""
        self.assertEqual(self.persona.id, "test-id")
        self.assertEqual(self.persona.name, "Test Persona")
        self.assertEqual(len(self.persona.traits), 3)
        self.assertIn("analytical", self.persona.traits)
        self.assertEqual(self.persona.background, "AI researcher with expertise in natural language processing")
        self.assertEqual(self.persona.communication_style, "clear and concise, with a focus on accuracy")
        self.assertEqual(self.persona.knowledge_areas, [])
        self.assertEqual(self.persona.additional_details, {})
    
    def test_generate_system_message(self):
        """Test that the system message is generated correctly."""
        message = self.persona._generate_system_message()
        
        # Check that the message contains the persona's attributes
        self.assertIn(self.persona.name, message)
        for trait in self.persona.traits:
            self.assertIn(trait, message)
        self.assertIn(self.persona.background, message)
        self.assertIn(self.persona.communication_style, message)
    
    @patch('src.persona_generator.api_client.ResponsesAPIClient')
    def test_generate_response(self, mock_api_client):
        """Test that generate_response calls the API client correctly."""
        # Setup mock
        mock_instance = MagicMock()
        mock_api_client.return_value = mock_instance
        mock_instance.generate_response.return_value = "This is a test response."
        
        # Call the method
        response = self.persona.generate_response("Test prompt")
        
        # Verify the result
        self.assertEqual(response, "This is a test response.")
        
        # Verify the API client was called correctly
        mock_instance.generate_response.assert_called_once()
        # The first arg should be the system message
        system_message = mock_instance.generate_response.call_args[1]['system_message']
        self.assertIn(self.persona.name, system_message)
        # The second arg should be the user message
        self.assertEqual(mock_instance.generate_response.call_args[1]['user_message'], "Test prompt")
    
    def test_to_dict(self):
        """Test that a persona can be converted to a dictionary."""
        persona_dict = self.persona.to_dict()
        
        self.assertEqual(persona_dict['id'], self.persona.id)
        self.assertEqual(persona_dict['name'], self.persona.name)
        self.assertEqual(persona_dict['traits'], self.persona.traits)
        self.assertEqual(persona_dict['background'], self.persona.background)
        self.assertEqual(persona_dict['communication_style'], self.persona.communication_style)
    
    def test_from_dict(self):
        """Test that a persona can be created from a dictionary."""
        persona_dict = {
            'id': 'dict-id',
            'name': 'Dict Persona',
            'traits': ['creative', 'thoughtful'],
            'background': 'Writer with a focus on science fiction',
            'communication_style': 'Imaginative and descriptive',
            'knowledge_areas': ['literature', 'writing'],
            'additional_details': {'favorite_genre': 'sci-fi'}
        }
        
        persona = Persona.from_dict(persona_dict)
        
        self.assertEqual(persona.id, 'dict-id')
        self.assertEqual(persona.name, 'Dict Persona')
        self.assertEqual(persona.traits, ['creative', 'thoughtful'])
        self.assertEqual(persona.background, 'Writer with a focus on science fiction')
        self.assertEqual(persona.communication_style, 'Imaginative and descriptive')
        self.assertEqual(persona.knowledge_areas, ['literature', 'writing'])
        self.assertEqual(persona.additional_details, {'favorite_genre': 'sci-fi'})


if __name__ == '__main__':
    unittest.main()
