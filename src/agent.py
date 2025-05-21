from abc import ABC, abstractmethod
import inspect
import random
from collections import Counter
import sys
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.action import Action

class Agent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def play(self,
        self_history: list[Action],
        opponent_history: list[Action],
    ) -> Action:
        """Decide what action to play this round."""
        pass

    def __str__(self):
        return self.name


class BaseAgent(Agent):
    def __init__(self, llm: BaseChatModel, name:str, rule: str):
        super().__init__(name)
        self.llm = llm
        self.rule = rule

    def play(self,
        self_history: list[Action],
        opponent_history: list[Action],
        *,
        reason: bool,
        self_consistency: int = 1,
    ) -> Action:

        assert len(self_history) == len(opponent_history), f"Self and opponent histories should have equal lengths, but got self:{len(self_history)} and opponent:{len(opponent_history)}"

        round_number = len(self_history) + 1

        prompt = f"Round {round_number}. Your past actions: {self_history}. Opponent past actions: {opponent_history}."
        if reason:
            prompt += """
            1. What strategy do you think your opponent is using?
            2. What is the best strategy to respond?
            3. You must say the final action as either \"I will choose C\" or \"I will choose D\"
            """
        else:
            prompt += 'You must say the final action as either \"I will choose C\" or \"I will choose D\"'

        messages = []
        # Mistral model does not accept System Message, so merge with user message
        if "mistral" in self.name.lower():
            combined_prompt = f"{self.rule.strip()}\n\n{prompt.strip()}"
            messages.append(HumanMessage(content=combined_prompt))
        else:
            messages = [
                SystemMessage(content=self.rule),
                HumanMessage(content=prompt)
            ]
        # print(self.name.lower())
        # print(messages)
        actions_picked = []
        reasons = []

        attempt = 0  # counts valid responses
        total_retries = 0  # counts all attempts including failed ones
        max_total_retries = 10 + self_consistency

        while attempt < self_consistency and total_retries < max_total_retries:
            total_retries += 1
            raw = self.llm.invoke(messages).content.strip()

            prompt_text = messages[-1].content.strip()
            response = raw.split(prompt_text)[-1].strip()
            reasons.append(response)

            pattern = r"I will choose\s*([CD])"
            all_choices = re.findall(pattern, response)

            if all_choices:
                last_choice = all_choices[-1]
                actions_picked.append(Action(last_choice))
                attempt += 1
            else:
                print(f"Retry {total_retries}: LLM failed at round {round_number}. Response:\n{response}", file=sys.stderr)

        if attempt < self_consistency:
            raise RuntimeError(f"LLM failed to produce {self_consistency} valid actions after {max_total_retries} retries.")

        # Majority vote for self consistency
        action = Counter(actions_picked).most_common(1)[0][0]
        return action, reasons


class CodeStrategyAgent(Agent):
    def __init__(self, llm: BaseChatModel, rule: str):
        self.llm = llm
        self.rule = rule

        self.strategy_text = self._generate_strategy()
        self.strategy_code = self._generate_code(self.strategy_text)
        self.play_fn = self._compile_play(self.strategy_code)

    def _generate_strategy(self) -> str:
        messages = [
            SystemMessage(content=self.rule),
            HumanMessage(content="""
            Develop a simple strategy for an iterated normal-form game that maximizes your overall payoff considering the game's payoff structure.
            Provide a straightforward description using only natural language with minimal commentary.
            Be clear and specific about the conditions governing when to cooperate or defect, and order them appropriately.
            """)
        ]

        strategy = self.llm.invoke(messages).content.strip()

        # Critique pass
        messages += [AIMessage(content=strategy), HumanMessage(content="Please list any logical issues or unclear parts in the strategy.")]
        critique = self.llm.invoke(messages).content.strip()

        # Refinement pass
        messages += [AIMessage(content=critique), HumanMessage(content="Please rewrite the strategy considering the feedback.")]
        refined_strategy = self.llm.invoke(messages).content.strip()

        return refined_strategy


    def _generate_code(self, strategy: str) -> str:
        return f"""Implement the following strategy description as an algorithm using python 3.11 and the Axelrod library.

        {strategy}

        {self.rule}

        Your response should only include the python code for the strategy function, which has the following signature:

        def play(self,
            self_history: list[Action],
            opponent_history: list[Action],
        ) -> Action:

        # Example:
        # self_history = [Action.COOPERATE, Action.DEFECT]
        # opponent_history = [Action.DEFECT, Action.COOPERATE]

        You use assume the following imports:

        import random
        import numpy as np

        {inspect.getsource(Action)}

        No other libraries are to be used and no additional member functions are to be defined, but you may create nested subfunctions.

        Begin your response by repeating the strategy function signature. Only include python code in your response.
        """

    def _compile_play(self, code: str):
        """Dynamically compile the LLM-generated play function."""
        namespace = {}
        try:
            exec(code, globals(), namespace)
            return namespace["play"]
        except Exception as e:
            raise RuntimeError(f"Failed to compile strategy code:\n{code}\n\nError: {e}")

    def play(
        self,
        self_history: list[Action],
        opponent_history: list[Action],
        **kwargs
    ) -> Action:
        """Call the generated strategy function."""
        return self.play_fn(self, self_history, opponent_history)