"""Central config, loaded from environment variables (.env supported via
python-dotenv). Every other file imports THIS file rather than reading
os.environ directly — one place to see all settings, one place to fix
if something's misconfigured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # reads .env and injects its values into os.environ


@dataclass
class Config:
    # LLM provider settings
    llm_provider: str = os.environ.get("LLM_PROVIDER", "groq")
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    groq_model: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    temperature: float = 0.0
    max_tokens: int = 2000

    # Search provider settings
    search_provider: str = os.environ.get("SEARCH_PROVIDER", "tavily")
    tavily_api_key: str = os.environ.get("TAVILY_API_KEY", "")
    search_top_k: int = 5

    logs_dir: str = "experiments/logs"

    # Adaptive loop bounds — prevents the agent loop from running forever
    max_retrieval_rounds: int = 3
    max_verification_rounds: int = 3

    def validate_for_run(self) -> list[str]:
        """Returns a list of problems (empty = ready to run)."""
        problems = []
        if self.llm_provider == "groq" and not self.groq_api_key:
            problems.append(
                "GROQ_API_KEY is not set in .env. Get a free key at "
                "https://console.groq.com"
            )
        if self.search_provider == "tavily" and not self.tavily_api_key:
            problems.append(
                "TAVILY_API_KEY is not set in .env. Get a free key at "
                "https://app.tavily.com"
            )
        return problems


DEFAULT_CONFIG = Config()