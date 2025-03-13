# Simulacra Web Interface

A web-based interface for the Simulacra document analysis and style replication tool, providing a user-friendly way to analyze documents, generate personas, and create content with specific writing styles.

## Features

- **Document Analysis**: Upload and analyze documents to extract writing style features and psychological traits
- **Persona Generation**: Create detailed author personas based on document analysis or manually
- **Style Replication**: Generate new content that authentically captures an author's writing style
- **Visualizations**: Interactive visualizations of persona features and writing characteristics
- **API Access**: Secured endpoints with JWT-based authentication

## Installation

1. Make sure you have Python 3.8+ installed

2. Clone this repository:
   ```
   git clone <repository-url>
   cd simulacra-web
   ```

3. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install required packages:
   ```
   pip install -r requirements.txt
   ```

5. Install the simulacra package:
   ```
   pip install -e ../simulacra
   ```

## Running the Application

Start the application using the provided run script:

```
./run.py
```

Or with Python directly:

```
python run.py
```

The web interface will be available at http://127.0.0.1:8000

## Usage

1. **Login/Register**: Create an account or login with the demo account (username: admin, password: secret)

2. **Document Analysis**:
   - Upload text documents (TXT, MD, DOCX, PDF)
   - View analysis results with interactive visualizations
   - Export detailed reports

3. **Persona Generation**:
   - Generate personas from document analysis
   - Create custom personas manually
   - Visualize persona traits

4. **Style Replication**:
   - Select a persona
   - Define content requirements
   - Adjust style parameters
   - Generate and refine content

## API Documentation

The API documentation is available at http://127.0.0.1:8000/docs once the server is running.

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (optional, can be provided per request)
- `SECRET_KEY`: Secret key for JWT token generation (defaults to a development key)

## License

[Specify the license information]
