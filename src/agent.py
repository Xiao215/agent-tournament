from abc import ABC, abstractmethod
import inspect
import random
from collections import Counter
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.action import Action

class Agent(ABC):
    @abstractmethod
    def play(self,
        self_history: list[Action],
        opponent_history: list[Action],
    ) -> Action:
        """Decide what action to play this round."""
        pass


class BaseAgent(Agent):
    def __init__(self, llm: BaseChatModel, rule: str):
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

        messages = [
            SystemMessage(content=self.rule)
        ]
        round_number = len(self_history) + 1

        prompt = f"Round {round_number}. Your past actions: {self_history}. Opponent past actions: {opponent_history}."
        if reason:
            prompt += """
            1. What strategy do you think your opponent is using?
            2. What is the best strategy to respond?
            3. Output your final action as either \"Final Answer: C\" or \"Final Answer: D\"
            """
        else:
            prompt += 'Output your final action as either \"Final Answer: C\" or \"Final Answer: D\"'
        messages.append(HumanMessage(content=prompt))

        actions_picked = []
        for _ in range(self_consistency):
            response = self.llm.invoke(messages).content.strip()
            match = re.search(r"Final Answer:\s*([CD])", response)
            if match:
                actions_picked.append(Action(match.group(1)))
            else:
                raise ValueError(
                    f'{self.llm} failed to make a action at round {round_number}, response was: {response}'
                )

        # Majority vote for self consistency
        action = Counter(actions_picked).most_common(1)[0][0]
        return action


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