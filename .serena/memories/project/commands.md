# Project Commands

## Setup
```bash
cd /home/soho/gitlab-repos/proxy-llm/llm4agents-sdk-python
source .venv/bin/activate
```

## Testing
```bash
# Full test suite
pytest tests/ -v

# Single test file
pytest tests/test_client.py -v

# Single test by name
pytest tests/test_client.py::test_client_creates -v

# With asyncio info
pytest --tb=short
```

## Code Quality
```bash
# Type checking (if mypy available)
mypy llm4agents --strict

# Linting (if ruff/flake8 available)
# Check project docs for configured linters
```

## Installation
```bash
# Install dev dependencies
pip install -e ".[dev]"
```

## Git Workflow
```bash
git add <files>
git commit -m "type: description"
git push origin <branch>
```
