# Vinkeljernet Refactoring

This document explains the refactoring of the Vinkeljernet application and how to use the new modular structure.

## Overview

The Vinkeljernet application has been refactored into a modular structure to improve maintainability, separation of concerns, and code organization. The main benefits are:

1. **Separation of Concerns**: Clear separation between CLI interface, core functionality, and UI components
2. **Improved Maintainability**: Smaller, more focused modules that are easier to maintain and understand
3. **Better Organization**: Logical grouping of related functionality
4. **Enhanced Testability**: Easier to write unit tests for specific components

## New Structure

The application is now organized into the following modules:

### Core Modules

- **vinkeljernet/cli.py**: Command-line interface and argument parsing
- **vinkeljernet/core.py**: Core functionality independent of the interface
- **vinkeljernet/interactive_mode.py**: Interactive CLI using prompt_toolkit
- **vinkeljernet/standard_mode.py**: Standard CLI mode processing
- **vinkeljernet/ui_utils.py**: Common UI utilities (progress bars, tables, panels)
- **vinkeljernet/utils.py**: General utility functions

### Entry Points

- **main_refactored.py**: Simple entry point script
- **vinkeljernet/__main__.py**: Package entry point

## Migration Guide

To migrate from the old structure to the new one:

1. **Installation**:
   ```
   # Create the vinkeljernet directory if it doesn't exist
   mkdir -p vinkeljernet
   
   # Copy the new files
   cp -r vinkeljernet/* /path/to/your/project/vinkeljernet/
   
   # Copy the new main script
   cp main_refactored.py /path/to/your/project/main.py
   ```

2. **Usage**:
   The application can still be run in the same way:
   ```
   python main.py [arguments]
   ```

3. **Development**:
   - To add new features, modify the appropriate module rather than adding to main.py
   - To add new commands to the interactive mode, modify interactive_mode.py
   - To modify the core angle generation logic, edit core.py

## Module Responsibilities

### cli.py
- Parse command-line arguments
- Dispatch to correct mode handler
- Handle top-level error handling

### core.py
- Generate angles based on topic and profile
- Process and rank angles
- API interactions independent of UI

### interactive_mode.py
- Interactive CLI interface 
- Command processing and history
- Interactive help and feedback

### standard_mode.py
- Non-interactive processing
- Output formatting for CLI mode

### ui_utils.py
- Progress bars and spinners
- Table formatting
- Panel and UI element construction

### utils.py
- File operations
- Configuration helpers
- Utility functions

## Testing

The modular structure makes it easier to write unit tests. For example:

```python
# Example test for the core module
import unittest
from vinkeljernet.core import safe_process_angles

class TestCore(unittest.TestCase):
    def test_safe_process_angles(self):
        # Test code here
        pass
```

## Future Improvements

With this modular structure in place, future improvements could include:

1. **Web Interface**: Add a web module that uses the same core functionality
2. **API Server**: Create an API server that exposes the core functionality as REST endpoints
3. **Plugin System**: Implement a plugin system for custom providers or formatters
4. **Extended Configuration**: Enhance the configuration system