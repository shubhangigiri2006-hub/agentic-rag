"""LLM provider factory.

get_llm(config) returns a LangChain chat model — something with a
.invoke(messages) method — built from whatever's in .env. Every agent node
calls this instead of talking to Groq directly, so provider-switching
later only touches this one file.
"""

from __future__ import annotations

from config import Config


def get_llm(config: Config):
    if config.llm_provider == "groq":
        from langchain_groq import ChatGroq

        if not config.groq_api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set in .env. Get a free key at "
                "https://console.groq.com"
            )
        return ChatGroq(
            model=config.groq_model,
            api_key=config.groq_api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    raise ValueError(
        f"Unknown llm_provider: {config.llm_provider!r} (only 'groq' is wired up so far)"
    )