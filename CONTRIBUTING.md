# Contributing to GRU Time Series Prediction

We welcome contributions to the GRU Time Series Prediction project! This document provides guidelines for contributing.

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/gru-time-series-prediction.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Install development dependencies: `pip install -e .[dev]`

## 🔧 Development Setup

### Pre-commit Hooks
Install pre-commit hooks to ensure code quality:
```bash
pre-commit install
```

### Code Formatting
We use Black for code formatting:
```bash
black .
```

### Linting
We use flake8 for linting:
```bash
flake8 .
```

## 📝 Code Style

- Follow PEP 8 guidelines
- Use Black for automatic formatting
- Maximum line length: 88 characters
- Use type hints where appropriate
- Write descriptive docstrings for functions and classes

## 🧪 Testing

### Running Tests
```bash
pytest
```

### Test Coverage
```bash
pytest --cov=. --cov-report=html
```

### Writing Tests
- Write tests for new features
- Maintain or improve test coverage
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern

## 📋 Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Commit with descriptive messages
7. Push to your fork
8. Create a pull request

### Pull Request Guidelines

- Provide a clear description of changes
- Reference any related issues
- Include screenshots for UI changes
- Ensure CI/CD pipeline passes
- Request review from maintainers

## 🐛 Bug Reports

When reporting bugs, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces
- Minimal code example

## 💡 Feature Requests

For feature requests, please:

- Check if the feature already exists
- Describe the use case
- Explain the expected behavior
- Consider implementation complexity
- Discuss alternatives

## 📚 Documentation

- Update README.md for significant changes
- Add docstrings to new functions/classes
- Update configuration documentation
- Include examples for new features

## 🏷️ Commit Messages

Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

Example: `feat: add attention mechanism to GRU model`

## 🔒 Security

- Report security vulnerabilities privately
- Don't include sensitive data in commits
- Use environment variables for secrets
- Follow security best practices

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on the code, not the person
- Help maintain a positive environment

## 📞 Getting Help

- Open an issue for questions
- Join discussions in pull requests
- Check existing documentation
- Review code examples

Thank you for contributing! 🎉
