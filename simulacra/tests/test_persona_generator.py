import unittest
from unittest.mock import patch, MagicMock

from src.persona_generator import PersonaGenerator, Persona


class TestPersonaGenerator(unittest.TestCase):
    """Test cases for the PersonaGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = PersonaGenerator(api_key="test-api-key")
    
    def test_initialization(self):
        """Test that a generator can be properly initialized."""
        self.assertEqual(self.generator.api_key, "test-api-key")
        self.assertEqual(self.generator.personas, {})
    
    def test_create_persona(self):
        """Test creating a new persona."""
        persona = self.generator.create_persona(
            name="Test Persona",
            traits=["creative", "analytical"],
            background="Technical writer with expertise in AI",
            communication_style="friendly and informative"
        )
        
        # Check the persona was created with the correct attributes
        self.assertIsInstance(persona, Persona)
        self.assertEqual(persona.name, "Test Persona")
        self.assertEqual(persona.traits, ["creative", "analytical"])
        self.assertEqual(persona.background, "Technical writer with expertise in AI")
        self.assertEqual(persona.communication_style, "friendly and informative")
        
        # Check the persona was stored in the generator
        self.assertEqual(len(self.generator.personas), 1)
        self.assertIn(persona.id, self.generator.personas)
        self.assertEqual(self.generator.personas[persona.id], persona)
    
    def test_get_persona(self):
        """Test retrieving a persona by ID."""
        # Create a persona
        persona = self.generator.create_persona(
            name="Test Persona",
            traits=["creative"],
            background="Background",
            communication_style="Style"
        )
        
        # Retrieve the persona
        retrieved_persona = self.generator.get_persona(persona.id)
        
        # Check it's the same persona
        self.assertEqual(retrieved_persona, persona)
        
        # Check retrieving a non-existent persona returns None
        self.assertIsNone(self.generator.get_persona("non-existent-id"))
    
    def test_update_persona(self):
        """Test updating a persona."""
        # Create a persona
        persona = self.generator.create_persona(
            name="Original Name",
            traits=["original"],
            background="Original background",
            communication_style="Original style"
        )
        
        # Update the persona
        updated_persona = self.generator.update_persona(
            persona.id,
            name="Updated Name",
            traits=["updated"],
            background="Updated background"
        )
        
        # Check the persona was updated
        self.assertEqual(updated_persona.name, "Updated Name")
        self.assertEqual(updated_persona.traits, ["updated"])
        self.assertEqual(updated_persona.background, "Updated background")
        # This field should not have changed
        self.assertEqual(updated_persona.communication_style, "Original style")
        
        # Check updating a non-existent persona returns None
        self.assertIsNone(self.generator.update_persona("non-existent-id", name="New Name"))
    
    def test_delete_persona(self):
        """Test deleting a persona."""
        # Create a persona
        persona = self.generator.create_persona(
            name="Test Persona",
            traits=["trait"],
            background="Background",
            communication_style="Style"
        )
        
        # Check the persona exists
        self.assertIn(persona.id, self.generator.personas)
        
        # Delete the persona
        result = self.generator.delete_persona(persona.id)
        
        # Check the result and that the persona was deleted
        self.assertTrue(result)
        self.assertNotIn(persona.id, self.generator.personas)
        
        # Check deleting a non-existent persona returns False
        self.assertFalse(self.generator.delete_persona("non-existent-id"))
    
    def test_list_personas(self):
        """Test listing all personas."""
        # Initially, there should be no personas
        self.assertEqual(len(self.generator.list_personas()), 0)
        
        # Create some personas
        persona1 = self.generator.create_persona(
            name="Persona 1",
            traits=["trait1"],
            background="Background 1",
            communication_style="Style 1"
        )
        
        persona2 = self.generator.create_persona(
            name="Persona 2",
            traits=["trait2"],
            background="Background 2",
            communication_style="Style 2"
        )
        
        # Check that list_personas returns both personas
        personas = self.generator.list_personas()
        self.assertEqual(len(personas), 2)
        self.assertIn(persona1, personas)
        self.assertIn(persona2, personas)
    
    @patch('openai_agents.Agent')
    def test_create_agent_from_persona(self, mock_agent):
        """Test creating an agent from a persona."""
        # Setup mock
        mock_agent.return_value = MagicMock()
        
        # Create a persona
        persona = self.generator.create_persona(
            name="Agent Persona",
            traits=["trait"],
            background="Background",
            communication_style="Style"
        )
        
        # Create an agent from the persona
        agent = self.generator.create_agent_from_persona(persona.id)
        
        # Check that the Agent constructor was called with the correct arguments
        mock_agent.assert_called_once()
        call_kwargs = mock_agent.call_args[1]
        self.assertEqual(call_kwargs['api_key'], "test-api-key")
        self.assertEqual(call_kwargs['model'], "gpt-4-turbo")
        self.assertIn(persona.name, call_kwargs['system_prompt'])
        
        # Check creating an agent from a non-existent persona returns None
        self.assertIsNone(self.generator.create_agent_from_persona("non-existent-id"))
    
    @patch('src.persona_generator.persona.Persona.generate_response')
    def test_simulate_conversation(self, mock_generate_response):
        """Test simulating a conversation between personas."""
        # Setup mock
        mock_generate_response.side_effect = ["Response 1", "Response 2"]
        
        # Create two personas
        persona1 = self.generator.create_persona(
            name="Persona 1",
            traits=["trait1"],
            background="Background 1",
            communication_style="Style 1"
        )
        
        persona2 = self.generator.create_persona(
            name="Persona 2",
            traits=["trait2"],
            background="Background 2",
            communication_style="Style 2"
        )
        
        # Simulate a conversation
        conversation = self.generator.simulate_conversation(
            persona1.id,
            persona2.id,
            "Initial message",
            num_turns=1
        )
        
        # Check the conversation structure
        self.assertEqual(len(conversation), 3)  # system + initial + 1 response
        self.assertEqual(conversation[0]['role'], "system")
        self.assertEqual(conversation[1]['role'], "user")
        self.assertEqual(conversation[1]['content'], "Persona 1: Initial message")
        self.assertEqual(conversation[2]['role'], "assistant")
        self.assertEqual(conversation[2]['content'], "Persona 2: Response 1")
        
        # Check that generate_response was called the correct number of times
        self.assertEqual(mock_generate_response.call_count, 1)
        
        # Check that simulating a conversation with non-existent personas raises an error
        with self.assertRaises(ValueError):
            self.generator.simulate_conversation("non-existent-id", persona2.id, "Message", 1)


if __name__ == '__main__':
    unittest.main()
