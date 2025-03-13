# Simulacra

![Simulacra Dashboard](Screenshot%202025-03-13%20at%2013.33.15.png)

Simulacra is an advanced AI agent toolkit for generating personas, analyzing documents, and replicating writing styles using OpenAI's APIs. The project consists of both a Python library and a web interface, making it accessible for developers and non-technical users alike.

## Features

### Core Library (simulacra/)
- **Persona Generation**: Create, manage, and simulate AI personas with specific traits and communication styles
- **Document Analysis**: Extract writing style features, vocabulary patterns, and psychological traits from document collections
- **Style Replication**: Generate content that authentically mimics specific writing styles
- **Agent Workflows**: Design complex agent-based workflows with multiple persona interactions

### Web Interface (simulacra-web/)
- **User-friendly Dashboard**: Access all features through an intuitive web interface
- **Document Upload & Analysis**: Upload documents in various formats (TXT, MD, DOCX, PDF)
- **Interactive Visualizations**: Explore persona traits and writing characteristics through visual representations
- **Content Management**: Generate, store, and edit content created with different personas
- **Secure API Access**: JWT-based authentication for programmatic access

## Repository Structure

```
simulacra/
├── simulacra/            # Core Python library
│   ├── src/              # Library source code
│   │   ├── agent_workflow.py
│   │   ├── document_analysis/
│   │   ├── file_operations/
│   │   ├── persona_generator/
│   │   └── style_replication/
│   ├── examples/         # Example scripts
│   ├── tests/            # Unit tests
│   └── config/           # Configuration files
│
├── simulacra-web/        # Web interface
│   ├── app/              # Flask web application
│   │   ├── api/          # API endpoints
│   │   ├── templates/    # HTML templates
│   │   └── static/       # Frontend assets
│   ├── analyses/         # Document analysis results
│   ├── personas/         # Generated personas
│   ├── generated_content/ # Content created by personas
│   └── uploads/          # Uploaded documents
│
└── README.md             # This documentation
```

## Installation

### Core Library

1. Clone the repository:
```bash
git clone https://github.com/yourusername/simulacra.git
cd simulacra
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r simulacra/requirements.txt
```

4. Set up your environment variables:
```bash
cp simulacra/config/example.env .env
# Edit .env with your API keys
```

### Web Interface

1. Install the core library (see above)

2. Install web interface dependencies:
```bash
pip install -r simulacra-web/requirements.txt
```

3. Install the simulacra package:
```bash
pip install -e ./simulacra
```

## Usage

### Using the Core Library

```python
from simulacra.persona_generator import PersonaGenerator
from simulacra.document_analysis import DocumentAnalysisAgent

# Generate a persona
generator = PersonaGenerator()
persona = generator.create_persona(
    name="Dr. Rachel Chen",
    traits=["analytical", "precise", "cautious"],
    background="Neuroscientist with 15 years of research experience",
    communication_style="formal and evidence-based"
)

# Generate content with the persona
response = persona.generate_response(
    prompt="What do you think about the latest advances in AI?"
)

# Analyze documents
agent = DocumentAnalysisAgent()
analysis = agent.analyze_documents(
    folder_path="path/to/document/folder",
    file_extensions=[".txt", ".md"]
)

# Generate a report
report_path = agent.generate_analysis_report(
    analysis=analysis,
    output_path="analysis_report.md"
)
```

### Running the Web Interface

Start the application:

```bash
cd simulacra-web
python run.py
```

The web interface will be available at http://127.0.0.1:8000

## Screenshots

### Dashboard
![Dashboard](Screenshot%202025-03-13%20at%2013.33.15.png)

### Document Analysis
![Document Analysis](Screenshot%202025-03-13%20at%2014.06.06.png)

### Persona Generation
![Persona Generation](Screenshot%202025-03-13%20at%2013.33.58.png)

### Style Replication
![Style Replication](Screenshot%202025-03-13%20at%2015.01.19.png)

## E-commerce Server

Simulacra also includes an experimental e-commerce server for demonstrating persona-based content generation in a retail context:

```bash
cd simulacra-web
python run_ecommerce_server.py
```

## Development

1. Set up the development environment:
```bash
pip install -e "simulacra[dev]"
```

2. Run tests:
```bash
cd simulacra
pytest
```

## License

MIT
