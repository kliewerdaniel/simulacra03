import os
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from simulacra.document_analysis import DocumentAnalysisAgent


class TestDocumentAnalysisAgent(unittest.TestCase):
    """Unit tests for the DocumentAnalysisAgent class."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample test files
        self.create_sample_files()
        
        # Initialize the agent with a mock API key
        self.agent = DocumentAnalysisAgent(
            api_key="mock_api_key",
            max_files_per_analysis=10
        )
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_dir)
    
    def create_sample_files(self):
        """Create sample text files for testing."""
        # Create subdirectories
        os.makedirs(os.path.join(self.test_dir, "subfolder"), exist_ok=True)
        
        # Sample content with different writing styles
        formal_content = """
        The implementation of artificial intelligence in modern healthcare systems represents
        a paradigm shift in medical diagnostics and treatment planning. Numerous studies have
        demonstrated the efficacy of machine learning algorithms in identifying patterns within
        complex medical data that might otherwise remain undetected by human practitioners.
        
        This technological advancement, while promising significant improvements in patient outcomes,
        nevertheless raises important ethical considerations regarding privacy, data security,
        and the appropriate balance between algorithmic and human decision-making processes in
        clinical settings.
        """
        
        informal_content = """
        So I was thinking about AI in healthcare the other day, and wow, it's really changing everything!
        Doctors can now use these cool algorithms to spot things they might miss on their own. It's pretty
        amazing how computers can learn to recognize patterns in all that medical data.
        
        But here's the thing - we've got to be careful about privacy and making sure we don't rely
        too much on the machines. I mean, do we really want computers making all the big decisions
        about our health? Probably not! There's definitely a balance to find here.
        """
        
        technical_content = """
        The neural network architecture implemented in our diagnostic system utilizes a convolutional
        base (ResNet-50) pretrained on ImageNet, followed by custom dense layers with dropout (p=0.3)
        to prevent overfitting. We achieved 94.3% accuracy (CI: 92.1-96.5%) on the validation dataset
        comprising 1,245 labeled medical images.
        
        Preprocessing included standard normalization, random rotations (±15°), and horizontal flips
        for data augmentation. Training was performed using Adam optimizer (learning rate: 1e-4)
        with early stopping (patience=10) based on validation loss.
        """
        
        # Create the files with different extensions
        with open(os.path.join(self.test_dir, "formal_document.txt"), "w") as f:
            f.write(formal_content)
            
        with open(os.path.join(self.test_dir, "informal_post.md"), "w") as f:
            f.write(informal_content)
            
        with open(os.path.join(self.test_dir, "subfolder", "technical_report.txt"), "w") as f:
            f.write(technical_content)
            
        # Create a non-text file that should be ignored
        with open(os.path.join(self.test_dir, "ignore_me.jpg"), "w") as f:
            f.write("This is not a real image file, just for testing file extension filtering")
    
    @patch('openai.OpenAI')
    def test_read_documents(self, mock_openai):
        """Test that documents are read correctly."""
        # Call the method
        documents = self.agent.read_documents(
            folder_path=self.test_dir,
            file_extensions=['.txt', '.md']
        )
        
        # Check results
        self.assertEqual(len(documents), 3)
        
        # Check that we have the right files
        file_names = [doc["name"] for doc in documents]
        self.assertIn("formal_document.txt", file_names)
        self.assertIn("informal_post.md", file_names)
        self.assertIn("technical_report.txt", file_names)
        
        # Check that non-text files are excluded
        self.assertNotIn("ignore_me.jpg", file_names)
    
    @patch('openai.OpenAI')
    def test_prepare_document_batches(self, mock_openai):
        """Test that documents are batched correctly."""
        # Set up some test documents
        self.agent.current_documents = [
            {
                "name": "doc1.txt",
                "content": "a" * 4000  # ~1000 tokens
            },
            {
                "name": "doc2.txt",
                "content": "b" * 4000  # ~1000 tokens
            },
            {
                "name": "doc3.txt",
                "content": "c" * 12000  # ~3000 tokens
            },
            {
                "name": "doc4.txt",
                "content": "d" * 20000  # ~5000 tokens
            }
        ]
        
        # Call the method with a batch size of 2000 tokens
        batches = self.agent._prepare_document_batches(max_batch_tokens=2000)
        
        # Check results
        self.assertEqual(len(batches), 3)  # Should be 3 batches
    
    @patch('openai.OpenAI')
    def test_analyze_documents_integration(self, mock_openai):
        """Integration test for the analyze_documents method."""
        # Mock OpenAI client and responses
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create mock response for features
        mock_features_response = MagicMock()
        mock_features_response.choices = [MagicMock()]
        mock_features_response.choices[0].message.content = """{
            "vocabulary_size": 500,
            "average_word_length": 5.2,
            "word_frequencies": {"the": 50, "and": 30},
            "rare_words": ["paradigm", "efficacy"],
            "average_sentence_length": 15.5,
            "sentence_length_variation": 3.2,
            "sentence_structures": {"simple": 5, "complex": 10},
            "idioms": [],
            "metaphors": [],
            "transition_phrases": ["nevertheless", "while"],
            "paragraph_structure": {},
            "punctuation_usage": {"period": 20, "comma": 30},
            "passive_voice_frequency": 0.3,
            "active_voice_frequency": 0.7,
            "document_count": 3,
            "total_word_count": 1500,
            "total_sentence_count": 100
        }"""
        
        # Create mock response for psychological traits
        mock_traits_response = MagicMock()
        mock_traits_response.choices = [MagicMock()]
        mock_traits_response.choices[0].message.content = """{
            "openness": 0.8,
            "conscientiousness": 0.7,
            "extraversion": 0.3,
            "agreeableness": 0.6,
            "neuroticism": 0.4,
            "formality_level": 0.9,
            "analytical_thinking": 0.85,
            "emotional_expressiveness": 0.2,
            "confidence_level": 0.75,
            "dominant_cognitive_patterns": ["analytical", "logical"],
            "communication_preferences": ["precise", "structured"],
            "thinking_style": "systematic"
        }"""
        
        # Create mock response for summary
        mock_summary_response = MagicMock()
        mock_summary_response.choices = [MagicMock()]
        mock_summary_response.choices[0].message.content = "This is a summary of the writing style."
        
        # Create mock response for characteristics
        mock_chars_response = MagicMock()
        mock_chars_response.choices = [MagicMock()]
        mock_chars_response.choices[0].message.content = """{"characteristics": [
            "Uses complex sentence structures",
            "Prefers formal language",
            "Employs technical vocabulary"
        ]}"""
        
        # Create mock response for recommendations
        mock_recs_response = MagicMock()
        mock_recs_response.choices = [MagicMock()]
        mock_recs_response.choices[0].message.content = """{"recommendations": [
            "Vary sentence length for improved readability",
            "Incorporate more metaphors",
            "Consider using more personal examples"
        ]}"""
        
        # Set up the mock completions.create method to return different responses
        mock_client.chat.completions.create.side_effect = [
            mock_features_response,      # First call for features
            mock_traits_response,        # Second call for psychological traits
            mock_summary_response,       # Third call for summary
            mock_chars_response,         # Fourth call for characteristics
            mock_recs_response           # Fifth call for recommendations
        ]
        
        # Run the analysis
        analysis = self.agent.analyze_documents(folder_path=self.test_dir)
        
        # Verify the analysis object
        self.assertEqual(analysis.features.vocabulary_size, 500)
        self.assertEqual(analysis.features.average_word_length, 5.2)
        self.assertEqual(analysis.psychological_profile.openness, 0.8)
        self.assertEqual(analysis.psychological_profile.thinking_style, "systematic")
        self.assertEqual(analysis.writing_style_summary, "This is a summary of the writing style.")
        
        # Verify that completions.create was called
        self.assertTrue(mock_client.chat.completions.create.called)
        
    @patch('openai.OpenAI')
    def test_merge_features(self, mock_openai):
        """Test that features are merged correctly."""
        feature_list = [
            {
                "vocabulary_size": 200,
                "average_word_length": 5.0,
                "word_frequencies": {"the": 20, "and": 15},
                "rare_words": ["paradigm", "efficacy"],
                "sentence_structures": {"simple": 3, "complex": 5}
            },
            {
                "vocabulary_size": 300,
                "average_word_length": 5.4,
                "word_frequencies": {"the": 30, "but": 10},
                "rare_words": ["algorithm", "efficacy"],
                "sentence_structures": {"simple": 2, "compound": 3}
            }
        ]
        
        merged = self.agent._merge_features(feature_list)
        
        # Check merged values
        self.assertEqual(merged["vocabulary_size"], 500)
        self.assertEqual(merged["average_word_length"], 5.2)
        self.assertEqual(merged["word_frequencies"]["the"], 50)
        self.assertEqual(len(merged["rare_words"]), 3)  # Duplicates removed
        self.assertEqual(merged["sentence_structures"]["simple"], 5)
        self.assertEqual(merged["sentence_structures"]["complex"], 5)
        self.assertEqual(merged["sentence_structures"]["compound"], 3)
        
    @patch('openai.OpenAI')
    def test_save_analysis(self, mock_openai):
        """Test that analysis is saved correctly."""
        # Create a mock analysis object
        from simulacra.document_analysis.document_analyzer import DocumentFeatures, PsychologicalProfile, AuthorAnalysis
        
        features = DocumentFeatures(
            vocabulary_size=500,
            average_word_length=5.2,
            word_frequencies={"the": 50, "and": 30},
            rare_words=["paradigm", "efficacy"],
            average_sentence_length=15.5,
            sentence_length_variation=3.2,
            sentence_structures={"simple": 5, "complex": 10},
            idioms=[],
            metaphors=[],
            transition_phrases=["nevertheless", "while"],
            paragraph_structure={},
            punctuation_usage={"period": 20, "comma": 30},
            passive_voice_frequency=0.3,
            active_voice_frequency=0.7,
            document_count=3,
            total_word_count=1500,
            total_sentence_count=100
        )
        
        profile = PsychologicalProfile(
            openness=0.8,
            conscientiousness=0.7,
            extraversion=0.3,
            agreeableness=0.6,
            neuroticism=0.4,
            formality_level=0.9,
            analytical_thinking=0.85,
            emotional_expressiveness=0.2,
            confidence_level=0.75,
            dominant_cognitive_patterns=["analytical", "logical"],
            communication_preferences=["precise", "structured"],
            thinking_style="systematic"
        )
        
        analysis = AuthorAnalysis(
            features=features,
            psychological_profile=profile,
            writing_style_summary="This is a test summary.",
            distinguishing_characteristics=["Trait 1", "Trait 2"],
            recommendations=["Recommendation 1", "Recommendation 2"]
        )
        
        # Save to a temporary file
        temp_file = os.path.join(self.test_dir, "analysis.json")
        saved_path = self.agent.save_analysis(analysis, temp_file)
        
        # Check that the file exists
        self.assertTrue(os.path.exists(saved_path))
        
        # Check that content is saved correctly
        import json
        with open(saved_path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data["features"]["vocabulary_size"], 500)
        self.assertEqual(saved_data["psychological_profile"]["openness"], 0.8)
        self.assertEqual(len(saved_data["recommendations"]), 2)


if __name__ == "__main__":
    unittest.main()
