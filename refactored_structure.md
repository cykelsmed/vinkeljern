# Vinkeljernet Refactored Structure

This document outlines the proposed modular structure for the Vinkeljernet application.

## 1. Module Structure

### Core Modules

1. **vinkeljernet/cli.py**
   - Command-line interface and argument parsing
   - Entry point for the application
   - Dispatch to appropriate mode (standard or interactive)

2. **vinkeljernet/core.py**
   - Core functionality independent of the interface
   - Main processing logic for generating angles

3. **vinkeljernet/interactive_mode.py**
   - Interactive CLI using prompt_toolkit
   - Command processing and interactive session management

4. **vinkeljernet/standard_mode.py**
   - Standard CLI mode processing
   - Non-interactive operation

5. **vinkeljernet/ui_utils.py**
   - Common UI utilities (progress bars, tables, panels)
   - Rich text formatting

6. **vinkeljernet/utils.py**
   - General utility functions
   - File and configuration helpers

### Entry Point

- **vinkeljernet/__main__.py**
  - Simple entry point to launch the application
  - Ensures proper import context

## 2. Dependencies Between Modules

```
               ┌─────────────────┐
               │   __main__.py   │
               └────────┬────────┘
                        │
                 ┌──────▼───────┐
                 │    cli.py    │
                 └──┬──────────┬┘
                    │          │
        ┌───────────▼─┐     ┌──▼────────────┐
        │standard_mode│     │interactive_mode│
        └──────┬──────┘     └────────┬───────┘
               │                     │
               │     ┌───────┐      │
               └────►│core.py│◄─────┘
                     └───┬───┘
                         │
               ┌─────────▼────────┐
               │     ui_utils.py   │
               └──────────────────┘
                         │
                         ▼
               ┌─────────────────┐
               │     utils.py    │
               └─────────────────┘
```

## 3. Module Responsibilities

### cli.py
- Parse command-line arguments
- Determine execution mode (standard or interactive)
- Initialize logging and environment
- Dispatch to correct mode handler

### core.py
- Generate angles based on topic and profile
- Process and rank angles
- Handle API interactions
- Process data independent of UI

### interactive_mode.py
- Interactive CLI interface 
- Command processing and history
- Session state management
- Interactive help and feedback

### standard_mode.py
- Non-interactive processing
- Output formatting for CLI mode
- Error handling specific to non-interactive mode

### ui_utils.py
- Progress bars and spinners
- Table formatting
- Panel and UI element construction
- Consistent UI helpers

### utils.py
- File operations
- Configuration helpers
- Utility functions
- Common data processing

## 4. Implementation Plan

### Phase 1: Initial Structure
1. Create the module files with proper imports
2. Move top-level code and global initialization
3. Implement basic module interconnections

### Phase 2: Function Migration
1. Move functions to their respective modules
2. Update import references
3. Ensure cross-module dependencies work

### Phase 3: Final Assembly
1. Create the entry point
2. Implement missing connectors
3. Test the complete flow

## 5. Function Allocation

### cli.py
- `parse_arguments()`
- `main()`

### core.py
- `main_async()`
- `process_generation_request()`
- `safe_process_angles()`

### interactive_mode.py
- `run_interactive_cli()`
- `display_welcome_message()`
- `display_help()`
- `get_available_profiles()`
- `get_profile_names()`

### standard_mode.py
- `run_simple_cli()`

### ui_utils.py
- Progress bar related functions 
- Table rendering functions
- Console output formatting

### utils.py
- File related functions
- Miscellaneous helper functions