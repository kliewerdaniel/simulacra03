# Simulacra: Agent-Based Persona Generator & Document Analyzer

Simulacra is a toolkit for working with AI agents using the OpenAI Agents SDK and Responses API. It provides tools for generating and managing AI personas as well as analyzing writing styles from document collections.

## Features

### Persona Generation
- Create and manage AI persona profiles
- Generate content based on persona attributes
- Simulate conversations between different personas
- Test personas against various scenarios

### Document Analysis
- Analyze writing style across a folder of documents
- Extract key stylistic features (vocabulary, sentence structure, idioms)
- Identify psychological traits based on writing patterns
- Generate comprehensive reports on author writing style

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kliewerdaniel/simulacra.git
cd simulacra
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the example environment file and fill in your API keys:
```bash
cp config/example.env .env
```

## Usage

### Using the Persona Generator

```python
from simulacra.persona_generator import PersonaGenerator

# Create a new persona generator
generator = PersonaGenerator()

# Generate a new persona
persona = generator.create_persona(
    name="Dr. Rachel Chen",
    traits=["analytical", "precise", "cautious"],
    background="Neuroscientist with 15 years of research experience",
    communication_style="formal and evidence-based"
)

# Generate content using the persona
response = persona.generate_response(
    prompt="What do you think about the latest advances in AI?"
)

print(response)
```

### Using the Document Analysis Agent

```python
from simulacra.document_analysis import DocumentAnalysisAgent

# Initialize the document analysis agent
agent = DocumentAnalysisAgent()

# Analyze a folder of documents
analysis = agent.analyze_documents(
    folder_path="path/to/document/folder",
    file_extensions=[".txt", ".md"]
)

# Generate a human-readable report
report_path = agent.generate_analysis_report(
    analysis=analysis,
    output_path="path/to/save/report.md"
)

print(f"Analysis report saved to: {report_path}")
```

You can also use the included command-line example script:

```bash
python -m simulacra.examples.document_analysis_example /path/to/documents --output-dir ./analysis_results
```

## Development

1. Set up the development environment:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest
```

## License

MIT
