"""Wrapper around IBM GENAI APIs for use in langchain"""
import logging
from typing import Any, List, Mapping, Optional

from pydantic import BaseModel, Extra

try:
    from langchain.llms.base import LLM
    from langchain.llms.utils import enforce_stop_tokens
except ImportError:
    raise ImportError("Could not import langchain: Please install langchain.")

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

from LLM.credentials import Credentials

logger = logging.getLogger(__name__)

__all__ = ["LangChainInterface"]


class LangChainInterface(LLM, BaseModel):
    """
    Wrapper around IBM watsonx.ai models.

        .. code-block:: python
            from ibm_watson_machine_learning.foundation_models import Model

            llm = LangChainInterface(model=ModelTypes.FLAN_UL2.value, params=generate_params, credentials=creds)
    """

    credentials: Credentials = None
    model: Optional[str] = None
    params: Optional[dict] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _params = self.params or GenParams()
        return {
            **{"model": self.model},
            **{"params": _params},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "IBM watsonx.ai"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the IBM watsonx.ai's inference endpoint.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python
                llm = LangChainInterface(model_id="google/flan-ul2", credentials=creds)
                response = llm("What is a molecule")
        """
        params = self.params or GenParams()

        wml_model = Model(
            model_id=self.model,
            params=params,
            credentials=self.credentials.wml_credentials,
            project_id=self.credentials.project_id
        )
        text = wml_model.generate_text(prompt, params)

        logger.info("Output of watsonx.ai call: {}".format(text))
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text