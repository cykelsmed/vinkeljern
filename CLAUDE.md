# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test, and Lint Commands

- Run all tests: `python run_tests.py`
- Run unit tests only: `python run_tests.py --unit`
- Run integration tests only: `python run_tests.py --integration`
- Run tests with coverage: `python run_tests.py --coverage`
- Run pytest directly: `pytest -v tests/unit/test_file.py::TestClass::test_function`
- Run lint check: `black . --check` and `isort . --check`
- Run type check: `mypy .`

## Code Style Guidelines

- Formatting: Black with 100 character line length
- Import sorting: isort with black profile
- Type annotations: Use for function parameters and return values
- Naming: snake_case for functions/variables, PascalCase for classes, ALL_CAPS for constants
- Docstrings: Triple quotes with descriptive text
- Error handling: Use specific exception types and decorators (@safe_execute, @retry_with_backoff)
- Module organization: Group imports (standard library, third-party, local)
- Testing: Use pytest with appropriate markers (unit, integration, slow)