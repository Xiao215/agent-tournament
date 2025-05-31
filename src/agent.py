from abc import ABC, abstractmethod
import inspect
import os
from datetime import datetime
from collections import Counter
import sys
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.action import Action
from config import OUTPUTS_DIR, CACHE_DIR

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
    def __init__(self,
        llm: BaseChatModel,
        name:str,
        rule: str,
        reasoning: bool = False
    ):
        super().__init__(f'{name}{"(reasoning)" if reasoning else ""}')
        self.llm = llm
        self.rule = rule
        self.reasoning = reasoning

    def play(self,
        self_history: list[Action],
        opponent_history: list[Action],
        *,
        self_consistency: int = 1,
    ) -> tuple[Action, str]:

        assert len(self_history) == len(opponent_history), f"Self and opponent histories should have equal lengths, but got self:{len(self_history)} and opponent:{len(opponent_history)}"

        round_number = len(self_history) + 1

        prompt = f"Round {round_number}. Your past actions: {self_history}. Opponent past actions: {opponent_history}."
        if self.reasoning:
            prompt += """
            1. Analyze your opponent's strategy based on their past actions.
            2. Decide the best response strategy for this round.
            3. End your response with the exact sentence:
                I will choose C
                or
                I will choose D
            """
        else:
            prompt += """\n End your response with the exact sentence:
            I will choose C
            or
            I will choose D

            Do not include any explanation, commentary, or additional text after that sentence.
            """

        messages = []
        # Mistral model does not accept System Message, so merge with user message
        if any(model in self.name for model in ("gemma-2-9b-it", "Mistral-7B-Instruct-v0.3")):
            combined_prompt = f"{self.rule.strip()}\n\n{prompt.strip()}"
            messages.append(HumanMessage(content=combined_prompt))
        else:
            messages = [
                SystemMessage(content=self.rule),
                HumanMessage(content=prompt)
            ]

        actions_picked = []
        reasons = []

        attempt = 0  # counts valid responses
        total_retries = 0  # counts all attempts including failed ones
        max_total_retries = 10 + self_consistency

        while attempt < self_consistency and total_retries < max_total_retries:
            total_retries += 1
            response = self.llm.invoke(messages).content.strip()

            pattern = r"I will choose\s*([CD])"
            all_choices = re.findall(pattern, response)

            if all_choices:
                reasons.append(response)
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
    def __init__(self, llm: BaseChatModel, name: str, rule: str, log: bool):
        super().__init__(f'{name}(code)')
        self.llm = llm
        self.rule = rule
        self.log = log

        self.play_fn = self._compile_play()

    def _generate_strategy(self) -> str:
        nlp_strategy_prompt = """
        Develop a simple strategy for an iterated normal-form game that maximizes your overall payoff considering the game's payoff structure.
        Provide a straightforward description using only natural language with minimal commentary.
        Be clear and specific about the conditions governing when to cooperate or defect, and order them appropriately.
        """

        messages = []
        # Mistral model does not accept System Message, so merge with user message
        if any(model in self.name for model in ("gemma-2-9b-it", "Mistral-7B-Instruct-v0.3")):
            combined_prompt = f"{self.rule.strip()}\n\n{nlp_strategy_prompt.strip()}"
            messages.append(HumanMessage(content=combined_prompt))
        else:
            messages = [
                SystemMessage(content=self.rule),
                HumanMessage(content=nlp_strategy_prompt)
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
        code_generation_prompt = f"""Implement the following strategy description as an algorithm using python 3.11

        {strategy}

        {self.rule}

        Implement only the following Python code—nothing else:

        1. The import for `Action` and the definition of `play(...)`, using this exact signature:

        from src.action import Action
        def play(
            self_history: list[str],
            opponent_history: list[str],
        ) -> Action:
            <YOUR STRATEGY HERE>

        2. The self_history and opponent_history are both a list of 'C' and 'D'. For example:
            ['C', 'D', 'D']

        3. You may assume numpy is installed. If you need it, you can import it as:
            import numpy as np

        4. You may assume `Action` is defined as:

        class Action(Enum):
            COOPERATE = "C"
            DEFECT    = "D"

        5. You may create nested helper functions inside `play`, but DO **NOT**:
        - Use any other libraries
        - Define any extra methods or classes
        - Wrap your answer in markdown or triple-backticks
        - Add any explanation or comments outside of your strategy code

        6. At the top of your response, repeat exactly:

        from src.action import Action
        def play(...

        and then provide only valid Python code.
        """

        return self.llm.invoke(code_generation_prompt).content.strip()

    def _compile_play(self):
        """Dynamically compile or load the LLM-generated play function."""
        namespace = {}
        try:
            # Hash the strategy text to create a unique cache filename
            file_path = CACHE_DIR / "code_strategy" / f"{self.name}.py"

            if file_path.exists():
                code = file_path.read_text(encoding='utf-8')
                source = 'cache'
            else:
                strategy = self._generate_strategy()
                code = self._generate_code(strategy)

                if self.log:
                    now = datetime.now()
                    date_path = now.strftime("%Y/%m/%d")
                    time_stamp = now.strftime("%H-%M")
                    dir_path = OUTPUTS_DIR / date_path
                    os.makedirs(dir_path, exist_ok=True)
                    self.log_file = open(dir_path / f"{self}_{time_stamp}.txt", "w", encoding="utf-8")
                    self.log_file.write(f"Strategy: {strategy}\n\n")
                    self.log_file.write(f"Code:\n{code}\n")

                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(code, encoding='utf-8')
                source = 'LLM'

            print(f'Code loaded from {source}: {code}')

            if code.startswith('```'):
                code = code.split('```', 2)[1]

            exec(code, globals(), namespace)

            return namespace['play']
        except Exception as e:
            raise RuntimeError(f"Failed to compile strategy code from {source}:\n{code}\n\nError: {e}") from e

    def play(
        self,
        self_history: list[Action],
        opponent_history: list[Action],
        **kwargs
    ) -> Action:
        return self.play_fn(self_history, opponent_history), 'No reasoning'
