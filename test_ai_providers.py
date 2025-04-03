"""
Test script for the AI providers abstraction layer.

This script tests the various AI providers implemented in ai_providers.py.
It tests both synchronous and asynchronous API calls for each provider.

Usage:
    python test_ai_providers.py
"""

import os
import sys
import asyncio
from typing import Dict, List, Any

# Make sure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add project root to path if needed
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ai_providers import (
    ProviderType,
    ProviderConfig,
    ProviderFactory,
    get_anthropic_provider,
    get_openai_provider,
    get_perplexity_provider,
    fetch_topic_information_async,
    generate_angles_with_provider
)

# Set to True to test OpenAI API
TEST_OPENAI = False

async def test_anthropic_async():
    """Test the asynchronous Anthropic/Claude API."""
    print("\n--- Testing Anthropic/Claude API (async) ---")
    provider = get_anthropic_provider()
    
    topic = "Klimaforandringer i Danmark"
    prompt = f"Giv mig en kort oversigt over den aktuelle situation vedrørende: {topic}. Max 3 sætninger."
    
    print(f"Sending prompt to Anthropic API: {prompt}")
    
    try:
        result = await provider.generate_async(prompt, max_tokens=200)
        print(f"Response (async):\n{result.content}")
        if result.usage:
            print(f"Token usage: {result.token_usage}")
        return True
    except Exception as e:
        print(f"Error testing Anthropic API: {e}")
        return False

def test_anthropic_sync():
    """Test the synchronous Anthropic/Claude API."""
    print("\n--- Testing Anthropic/Claude API (sync) ---")
    provider = get_anthropic_provider()
    
    topic = "Klimaforandringer i Danmark"
    prompt = f"Giv mig en kort oversigt over den aktuelle situation vedrørende: {topic}. Max 3 sætninger."
    
    print(f"Sending prompt to Anthropic API: {prompt}")
    
    try:
        result = provider.generate(prompt, max_tokens=200)
        print(f"Response (sync):\n{result.content}")
        if result.usage:
            print(f"Token usage: {result.token_usage}")
        return True
    except Exception as e:
        print(f"Error testing Anthropic API: {e}")
        return False

async def test_perplexity_async():
    """Test the asynchronous Perplexity API."""
    print("\n--- Testing Perplexity API (async) ---")
    provider = get_perplexity_provider()
    
    topic = "Klimaforandringer i Danmark"
    prompt = f"Giv mig en kort oversigt over den aktuelle situation vedrørende: {topic}. Max 3 sætninger."
    
    print(f"Sending prompt to Perplexity API: {prompt}")
    
    try:
        result = await provider.generate_async(prompt, max_tokens=200)
        print(f"Response (async):\n{result.content}")
        if result.usage:
            print(f"Token usage: {result.token_usage}")
        return True
    except Exception as e:
        print(f"Error testing Perplexity API: {e}")
        return False

def test_perplexity_sync():
    """Test the synchronous Perplexity API."""
    print("\n--- Testing Perplexity API (sync) ---")
    provider = get_perplexity_provider()
    
    topic = "Klimaforandringer i Danmark"
    prompt = f"Giv mig en kort oversigt over den aktuelle situation vedrørende: {topic}. Max 3 sætninger."
    
    print(f"Sending prompt to Perplexity API: {prompt}")
    
    try:
        result = provider.generate(prompt, max_tokens=200)
        print(f"Response (sync):\n{result.content}")
        if result.usage:
            print(f"Token usage: {result.token_usage}")
        return True
    except Exception as e:
        print(f"Error testing Perplexity API: {e}")
        return False

async def test_openai_async():
    """Test the asynchronous OpenAI API."""
    if not TEST_OPENAI:
        print("\n--- Skipping OpenAI API test (async) ---")
        return True
        
    print("\n--- Testing OpenAI API (async) ---")
    provider = get_openai_provider()
    
    topic = "Klimaforandringer i Danmark"
    prompt = f"Giv mig en kort oversigt over den aktuelle situation vedrørende: {topic}. Max 3 sætninger."
    
    print(f"Sending prompt to OpenAI API: {prompt}")
    
    try:
        result = await provider.generate_async(prompt, max_tokens=200)
        print(f"Response (async):\n{result.content}")
        if result.usage:
            print(f"Token usage: {result.token_usage}")
        return True
    except Exception as e:
        print(f"Error testing OpenAI API: {e}")
        return False

def test_openai_sync():
    """Test the synchronous OpenAI API."""
    if not TEST_OPENAI:
        print("\n--- Skipping OpenAI API test (sync) ---")
        return True
        
    print("\n--- Testing OpenAI API (sync) ---")
    provider = get_openai_provider()
    
    topic = "Klimaforandringer i Danmark"
    prompt = f"Giv mig en kort oversigt over den aktuelle situation vedrørende: {topic}. Max 3 sætninger."
    
    print(f"Sending prompt to OpenAI API: {prompt}")
    
    try:
        result = provider.generate(prompt, max_tokens=200)
        print(f"Response (sync):\n{result.content}")
        if result.usage:
            print(f"Token usage: {result.token_usage}")
        return True
    except Exception as e:
        print(f"Error testing OpenAI API: {e}")
        return False

async def test_provider_factory():
    """Test the provider factory."""
    print("\n--- Testing Provider Factory ---")
    
    try:
        # Test creating an Anthropic provider
        provider = ProviderFactory.create_provider(ProviderType.ANTHROPIC)
        assert provider.__class__.__name__ == "AnthropicProvider"
        print("✓ Successfully created AnthropicProvider")
        
        # Test creating a Perplexity provider
        provider = ProviderFactory.create_provider(ProviderType.PERPLEXITY)
        assert provider.__class__.__name__ == "PerplexityProvider"
        print("✓ Successfully created PerplexityProvider")
        
        # Test creating an OpenAI provider
        provider = ProviderFactory.create_provider(ProviderType.OPENAI)
        assert provider.__class__.__name__ == "OpenAIProvider"
        print("✓ Successfully created OpenAIProvider")
        
        # Test with custom config
        config = ProviderConfig(
            api_key="fake_key_for_testing",
            model="custom-model",
            temperature=0.5
        )
        provider = ProviderFactory.create_provider(ProviderType.ANTHROPIC, config)
        assert provider.config.api_key == "fake_key_for_testing"
        assert provider.config.model == "custom-model"
        assert provider.config.temperature == 0.5
        print("✓ Successfully created provider with custom config")
        
        return True
    except Exception as e:
        print(f"Error testing Provider Factory: {e}")
        return False

async def test_backward_compatibility():
    """Test backward compatibility functions."""
    print("\n--- Testing Backward Compatibility ---")
    
    try:
        # Test fetch_topic_information_async
        topic = "Klimaforandringer i Danmark"
        print(f"Testing fetch_topic_information_async with topic: {topic}")
        
        # Progress callback for testing
        async def progress_callback(percent):
            print(f"Progress: {percent}%")
        
        result = await fetch_topic_information_async(
            topic=topic,
            dev_mode=False,
            bypass_cache=True,
            progress_callback=progress_callback
        )
        
        if result:
            print(f"Successfully fetched information (preview):\n{result[:200]}...")
            
        return True
    except Exception as e:
        print(f"Error testing backward compatibility: {e}")
        return False

async def run_all_tests():
    """Run all tests."""
    print("=== Running AI Provider Tests ===")
    
    results = {}
    
    # Test provider factory first
    results["provider_factory"] = await test_provider_factory()
    
    # Test Anthropic
    results["anthropic_sync"] = test_anthropic_sync()
    results["anthropic_async"] = await test_anthropic_async()
    
    # Test Perplexity
    results["perplexity_sync"] = test_perplexity_sync()
    results["perplexity_async"] = await test_perplexity_async()
    
    # Test OpenAI (optional)
    results["openai_sync"] = test_openai_sync()
    results["openai_async"] = await test_openai_async()
    
    # Test backward compatibility
    results["backward_compatibility"] = await test_backward_compatibility()
    
    # Print summary
    print("\n=== Test Results ===")
    all_passed = True
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        if not result:
            all_passed = False
        print(f"{test_name}: {status}")
    
    print(f"\nOverall result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed

if __name__ == "__main__":
    asyncio.run(run_all_tests())