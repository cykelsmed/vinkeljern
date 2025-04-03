import json
import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from models import VinkelForslag

logger = logging.getLogger(__name__)

def robust_json_parse(response_text: str, context: str = "response") -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Robust parser for extracting JSON from API responses with multiple fallback strategies.
    
    Args:
        response_text: Raw text response from LLM that should contain JSON
        context: Context description for error messages (e.g., "Claude response")
        
    Returns:
        Tuple containing:
        - List of parsed JSON objects
        - Error message if parsing ultimately failed, None if successful
    """
    # Setup result tracking
    results = []
    error_message = None
    
    # Strategy 1: Direct JSON parsing
    try:
        parsed_data = json.loads(response_text)
        
        # Handle both direct list or object with list property
        if isinstance(parsed_data, list):
            results = parsed_data
        elif isinstance(parsed_data, dict) and "vinkler" in parsed_data:
            results = parsed_data["vinkler"]
        elif isinstance(parsed_data, dict) and "angles" in parsed_data:
            results = parsed_data["angles"]
        elif isinstance(parsed_data, dict) and "results" in parsed_data:
            results = parsed_data["results"]
        elif isinstance(parsed_data, dict):
            # Single object case
            results = [parsed_data]
            
        logger.info(f"Successfully parsed JSON directly: {len(results)} items found")
        return results, None
    except json.JSONDecodeError as e:
        logger.warning(f"Direct JSON parsing failed: {e}")
        error_message = f"Direct JSON parsing failed: {str(e)}"
    
    # Strategy 2: Extract JSON from markdown code blocks
    json_blocks = []
    code_block_patterns = [
        r"```json\s*([\s\S]*?)\s*```",  # ```json ... ``` format
        r"```\s*([\s\S]*?)\s*```",      # ``` ... ``` format (any code block)
        r"`\s*([\s\S]*?)\s*`"           # ` ... ` format (inline code)
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, response_text)
        if matches:
            json_blocks.extend(matches)
    
    if json_blocks:
        for i, block in enumerate(json_blocks):
            try:
                parsed_data = json.loads(block)
                
                # Same handling as above
                if isinstance(parsed_data, list):
                    results = parsed_data
                elif isinstance(parsed_data, dict) and any(key in parsed_data for key in ["vinkler", "angles", "results"]):
                    for key in ["vinkler", "angles", "results"]:
                        if key in parsed_data:
                            results = parsed_data[key]
                            break
                else:
                    results = [parsed_data]
                    
                logger.info(f"Successfully parsed JSON from code block {i+1}: {len(results)} items found")
                return results, None
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Fix common JSON format issues
    try:
        # 3.1: Try to find JSON-like structures with regex
        potential_json = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
        if potential_json:
            try:
                fixed_json = potential_json.group(0)
                results = json.loads(fixed_json)
                logger.info(f"Successfully parsed JSON after regex extraction: {len(results)} items found")
                return results, None
            except json.JSONDecodeError:
                pass
        
        # 3.2: Replace common formatting issues and try again
        fixes = [
            (r'(\w+):', r'"\1":'),          # Fix unquoted keys
            (r"'([^']*)'", r'"\1"'),        # Replace single quotes with double quotes
            (r',\s*}', '}'),                # Remove trailing commas in objects
            (r',\s*\]', ']'),               # Remove trailing commas in arrays
            (r'True', 'true'),              # Fix Python True/False/None
            (r'False', 'false'),
            (r'None', 'null'),
        ]
        
        fixed_text = response_text
        for pattern, replacement in fixes:
            fixed_text = re.sub(pattern, replacement, fixed_text)
        
        try:
            parsed_data = json.loads(fixed_text)
            if isinstance(parsed_data, list):
                results = parsed_data
            elif isinstance(parsed_data, dict):
                results = [parsed_data]
                
            logger.info("Successfully parsed JSON after applying fixes")
            return results, None
        except json.JSONDecodeError:
            pass
    except Exception as e:
        logger.warning(f"Error during JSON recovery attempts: {e}")
    
    # Strategy 4: If we found any text that looks like JSON structures, make a last attempt
    try:
        # Find all sections that look like JSON objects
        object_pattern = r'{[^{}]*(?:{[^{}]*}[^{}]*)*}'
        potential_objects = re.findall(object_pattern, response_text)
        
        if potential_objects:
            valid_objects = []
            for obj_text in potential_objects:
                try:
                    obj = json.loads(obj_text)
                    valid_objects.append(obj)
                except:
                    continue
            
            if valid_objects:
                logger.info(f"Extracted {len(valid_objects)} partial JSON objects")
                return valid_objects, "Partial parsing succeeded, but some data may be missing"
    except Exception as e:
        logger.warning(f"Error during partial object extraction: {e}")
    
    # If all strategies failed
    detailed_error = f"""
    JSON Parsing Error in {context}:
    1. Direct parsing failed
    2. Code block extraction failed
    3. Recovery attempts failed
    
    Original error: {error_message}
    
    First 100 chars of response: {response_text[:100]}...
    """
    logger.error(detailed_error)
    
    return [], detailed_error.strip()

def parse_angles_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Analyzes LLM response and converts to a list of angle objects with improved error handling.
    
    Args:
        response_text: Raw text response from LLM
        
    Returns:
        List[Dict]: List of structured angle objects
    """
    angles = []
    
    # Try to parse response using robust parser
    parsed_data, error = robust_json_parse(response_text, "angle generation response")
    
    if not parsed_data:
        logger.error(f"Failed to parse angles from LLM response: {error}")
        return []
    
    # Process each angle
    for raw_angle in parsed_data:
        try:
            # Validate with Pydantic model
            validated_angle = VinkelForslag(**raw_angle).dict()
            angles.append(validated_angle)
        except Exception as e:
            # Log the raw angle that caused the issue for better debugging
            angle_preview = str(raw_angle)[:100] + "..." if len(str(raw_angle)) > 100 else str(raw_angle)
            logger.warning(f"Skipping invalid angle ({str(e)}): {angle_preview}")
            continue
    
    logger.info(f"Successfully parsed and validated {len(angles)} angles from response")
    return angles