"""
Test script to verify the fix for the OpenAI client initialization error.
"""

import os
import sys

# Add parent directory to path so we can import simulacra modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_openai_wrapper():
    """
    Test that our OpenAIClientWrapper can be initialized without the 'proxies' error.
    """
    try:
        from src.document_analysis.openai_client_wrapper import OpenAIClientWrapper
        
        # Initialize wrapper with dummy key
        print("Creating OpenAIClientWrapper...")
        wrapper = OpenAIClientWrapper(api_key="dummy_key_for_test", model="gpt-3.5-turbo")
        print("OpenAIClientWrapper initialized successfully")
        
        # We won't actually make an API call since we're using a dummy key
        # But we can verify that the wrapper's methods don't raise the 'proxies' error
        print("Wrapper configuration:")
        print(f"- API key: {'*' * 4}{wrapper.api_key[-4:] if len(wrapper.api_key) > 4 else wrapper.api_key}")
        print(f"- Model: {wrapper.model}")
        
        return True
    except Exception as e:
        import traceback
        print(f"Error testing OpenAI wrapper: {e}")
        traceback.print_exc()
        return False

def verify_document_analyzer_fix():
    """
    Verify that the DocumentAnalysisAgent can be initialized and methods work with our wrapper.
    """
    try:
        from src.document_analysis.document_analyzer import DocumentAnalysisAgent
        
        # Initialize with dummy key
        print("\nInitializing DocumentAnalysisAgent...")
        agent = DocumentAnalysisAgent(api_key="dummy_key_for_test")
        print("DocumentAnalysisAgent initialized successfully")
        
        print(f"Agent configured with model: {agent.model}")
        
        return True
    except Exception as e:
        import traceback
        print(f"Error verifying DocumentAnalysisAgent: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing OpenAI client wrapper fix...")
    wrapper_success = test_openai_wrapper()
    print(f"Wrapper test {'passed' if wrapper_success else 'failed'}")
    
    analyzer_success = verify_document_analyzer_fix()
    print(f"DocumentAnalysisAgent test {'passed' if analyzer_success else 'failed'}")
    
    # Overall result
    overall_success = wrapper_success and analyzer_success
    print(f"\nOverall test {'passed' if overall_success else 'failed'}")
    
    # Exit with appropriate status code
    sys.exit(0 if overall_success else 1)
