from abc import ABC, abstractmethod
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging as hf_logging
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.chat_models import ChatHuggingFace

from config import MODEL_WEIGHTS_DIR

# Suppress HF warnings
hf_logging.set_verbosity_error()

class LLMInstance():
    """A class to manage a Hugging Face LLM pipeline that can be moved between CPU and GPU."""
    def __init__(self, model_name: str):
        self.model_path = MODEL_WEIGHTS_DIR / model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,      # use float16 for better performance on GPU
            device_map="auto",              # automatically shards layers across GPU/CPU
            offload_folder="hf_offload",
            offload_state_dict=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

class LLMManager():
    """A class to manage a Hugging Face LLM pipeline that can be moved between CPU and GPU."""
    def __init__(self) -> None:
        self.llms = dict()

    def get_pipeline(self, model_name: str, **kwargs: Any) -> ChatHuggingFace:
        """Get an LLM instance for the given model name."""
        if model_name not in self.llms:
            self.llms[model_name] = LLMInstance(model_name)

        llm = self.llms[model_name]
        pipe = pipeline(
            "text-generation",
            model=llm.model,
            tokenizer=llm.tokenizer,
            return_full_text=False,        # only returns new tokens
            **kwargs
        )
        hf_pipe = HuggingFacePipeline(pipeline=pipe)
        chat_pipe = ChatHuggingFace(llm=hf_pipe, model_id=str(llm.model_path))
        return chat_pipe


class Agent(ABC):
    """
    Abstract base class for an LLM-based agent.
    """
    llm_manager = LLMManager()

    def __init__(self, llm_config: dict) -> None:
        self.name = llm_config['model']
        kwargs = llm_config.get('kwargs', {})
        self.chat_pipe = type(self).llm_manager.get_pipeline(self.name, **kwargs)

    @abstractmethod
    def chat(self,
        messages: str,
    ) -> str:
        """Chat with the agent using the provided messages."""
        raise NotImplementedError

    def invoke(self, messages: str) -> str:
        """Invoke the agent using the provided messages. No prompting added."""
        return self.chat_pipe.invoke(messages).content.strip()

    def __str__(self):
        return f"{self.name}({self.__class__.__name__})"

class IOAgent(Agent):
    """Input/Output Agent.
    This agent is designed to be the most basic llm agent. Given a message, answer it.
    """

    def chat(self,
        messages: str,
    ) -> str:
        """Chat with the agent using the provided messages."""
        messages += (
            "\nPlease ONLY provide the action you want to take, for example: <Action1>.\n"
            "DO NOT provide any additional text or explanation.\n"
            "Action:"
        )
        response = self.chat_pipe.invoke(messages)
        return response.content.strip()

class CoTAgent(Agent):
    """Chain-of-Thought Agent.

    This agent wraps the prompt to ask the LLM to think step-by-step.
    """

    def chat(self,
        messages: str,
    ) -> str:
        """Chat with the agent using the provided messages."""
        messages += """
        First reason about the strategy you are taking **step by step, then you must choose one action from legal actions.

        Your output must be in the following format strictly:

        Thought:
        Your thought.

        Action:
        Your action wrapped in angles brackets, for example: <Action1>
        """

        response = self.chat_pipe.invoke(messages)
        return response.content.strip()
