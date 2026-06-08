import logging
import os
from typing import Any, List

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from pydantic import SecretStr

from llms.Llm import Llm


class ClaudeLlm(Llm):
    """
    Concrete implementation of the Llm abstract class for interacting with Anthropic's
    Claude models (via the ChatAnthropic LangChain wrapper).

    This class handles model-specific configuration, API key management, token limits,
    web search integration, and response cleanup.
    """

    __DEFAULT_TOKEN_LIMIT = 200_000

    # Model metadata, including token limits and aliases
    __MODELS = {
        "claude-opus-4-8": {"aliases": ["claude-opus", "claude-4.8"]},
        "claude-opus-4-6": {},
        "claude-sonnet-4-6": {"aliases": ["claude-sonnet", "claude"]},
        "claude-sonnet-4-5-20250929": {"aliases": ["claude-4.5"]},
        "claude-haiku-4-5-20251001": {"aliases": ["claude-haiku"]},
        "claude-sonnet-4-20250514": {"aliases": ["claude-4"]},
        "claude-3-7-sonnet-20250219": {"aliases": ["claude-3.7"]},
        "claude-3-5-sonnet-20241022": {"aliases": ["claude-3.5"]},
        "claude-3-haiku-20240307": {"aliases": ["claude-3-haiku"]},
    }

    # List of all canonical model names
    SUPPORTED_MODELS = list(__MODELS.keys())

    # Models that support web search via Anthropic's built-in tool
    WEB_SEARCH_SUPPORTED = [
        "claude-opus-4-8",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
    ]

    # Mapping from aliases (e.g., 'claude') to canonical names
    MODEL_ALIASES = Llm._alias2model(__MODELS)

    # Dictionary mapping canonical model names to their context window size
    __MODEL_TOKEN_LIMITS = Llm._model_token_limit(__MODELS, __DEFAULT_TOKEN_LIMIT)

    __WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

    def __init__(
            self,
            model_name: str = "claude-sonnet",
            model_key: str | None = None,
            web_search: bool = False,
            **kwargs):
        """
        Initializes the Claude LLM client.

        Args:
            model_name: The requested model name or alias. Defaults to "claude-sonnet".
            model_key: The Anthropic API key. Searches environment variable if None.
            web_search: If True, binds Anthropic web search tool to supported models.
            **kwargs: Additional parameters passed directly to ChatAnthropic.

        Raises:
            ValueError: If the model is not supported.
            RuntimeError: If the API key is not found.
            NotImplementedError: If web search is requested for an unsupported model.
        """
        self.model_name = model_name

        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"LLM model {model_name} not supported")

        logging.info(f"Using {self.model_name}")

        self.model_key = model_key if model_key else os.environ.get("ANTHROPIC_API_KEY", None)
        if not self.model_key:
            raise RuntimeError("Anthropic API key not provided")

        self.llm_model = ChatAnthropic(
            model_name=self.model_name,
            api_key=SecretStr(self.model_key),
            **kwargs
        )

        if web_search:
            if self.model_name not in self.WEB_SEARCH_SUPPORTED:
                raise NotImplementedError(f"Web search is not supported by {self.model_name}")
            self.llm = self.llm_model.bind_tools([self.__WEB_SEARCH_TOOL])
        else:
            self.llm = self.llm_model

        super().__init__(llm=self.llm)

    def clean_up_response(self, response: Any) -> Llm.Response:
        """
        Cleans up the raw response from the LangChain wrapper.

        Args:
            response: The raw output from the ChatAnthropic runnable.

        Returns:
            A dictionary containing the cleaned response content and metadata.

        Raises:
            TypeError: If the response object is not the expected AIMessage.
        """
        if isinstance(response, AIMessage):
            return Llm.Response(
                text=response.text,
                raw=response,
            )

        else:
            raise TypeError(f"Unsupported return type for ClaudeLlm.invoke() (was {type(response)})")

    def get_num_tokens(self, text: str) -> int:
        """Estimates the number of tokens in the given text."""
        return super().get_num_tokens(text)

    def get_max_tokens(self) -> int:
        """Returns the maximum context window size for the currently selected model."""
        return self.__MODEL_TOKEN_LIMITS.get(self.model_name, 100_000)

    def get_model_name(self) -> str:
        return self.model_name

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Returns all supported model names (canonical names and aliases)."""
        return list(cls.MODEL_ALIASES.keys()) + cls.SUPPORTED_MODELS

    def as_runnable(self) -> Runnable:
        """Returns the underlying ChatAnthropic instance as a LangChain Runnable."""
        return self.llm

    def as_language_model(self) -> BaseLanguageModel:
        """Returns the underlying ChatAnthropic instance as a LangChain BaseLanguageModel."""
        return self.llm_model
