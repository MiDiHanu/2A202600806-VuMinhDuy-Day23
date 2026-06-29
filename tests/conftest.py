"""Pytest configuration.

Load environment variables from .env at collection time so that LLM-dependent
tests (which gate on GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY in their
skip markers) can detect a configured key before those markers are evaluated.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
