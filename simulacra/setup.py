from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="simulacra",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Agent-based persona generator using OpenAI Agents SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/simulacra",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements + [
        # Additional dependencies for file operations
        "PyPDF2>=3.0.0",  # PDF parsing
        "python-docx>=0.8.11",  # DOCX parsing
        "striprtf>=0.0.22",  # RTF parsing
        "PyYAML>=6.0",  # YAML serialization
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
)
