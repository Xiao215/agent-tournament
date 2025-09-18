"""Mechanism that lets players propose and sign payoff-altering contracts."""

import json
import re
import textwrap
from typing import Sequence

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.games.base import Game
from src.logger_manager import LOGGER
from src.mechanisms.base import Mechanism
from src.utils.concurrency import run_tasks


class Contracting(Mechanism):
    """Mechanism where players negotiate and optionally sign payoff contracts."""

    def __init__(
        self,
        base_game: Game,
    ) -> None:
        super().__init__(base_game)
        self.contracts: dict[str, list[int]] = {}
        self.contracts_design_prompt = textwrap.dedent(
            """
            Instruction:
            You are tasked with designing a **contract** for this game.
            - A contract is an extra payoff agreement.
            - Each player may choose to sign it or not.
            - The contract is valid only if all players sign.
            - For each action:
                * Positive number (+): the player receives X payoff from others.
                * Negative number (-): the player pays Y payoff to others.
                * Zero (0): no extra payoff.
            - Goal: maximize the total payoff for all players if the contract is signed.

            Output Requirement:
            - Return exactly **one valid Python dictionary** on a single line.
            - Format: {"A0": <INT>, "A1": <INT>, ...}
            - Keys: all available game actions.
            - Values: integers representing the extra payoff for that action.
            - Ensure the dictionary is syntactically valid Python.
            """
        )

        self.contract_confirmation_prompt = textwrap.dedent(
            """
            Contract Rule:
            On top of the original game instructions, you have the option to sign a contract.
            A contract is an extra payoff agreement that is valid only if all players sign.
            Here is the contract:
            {contract_description}

            Output Requirement:
            - Respond with a valid JSON object.
            - Format: {"sign": <BOOL>} where <BOOL> is true or false.
            """
        )

        self.contract_mechanism_prompt = textwrap.dedent(
            """
            Contract Rule:
            On top of the original game instructions, everyone has agreed to sign a contract.
            Here is the contract:
            {contract_description}

            Since this contract directly change your final payoff, consider the contract when making your decision for the strategy!
            """
        )

    def _design_contract(self, designer: Agent) -> tuple[str, list[int]]:
        """
        Design a contract from the given LLM agent.

        Returns:
            response (str): The raw response from the designer.
            contract (dict[int]): The contract with index representing the action
                and value representing the payoff adjustment.
        """
        base_prompt = (
            self.base_game.prompt + "\n" + self.contracts_design_prompt.format()
        )
        response, contract = designer.chat_with_retries(
            base_prompt=base_prompt,
            parse_func=self._parse_contract,
        )
        return response, contract

    def _agree_to_contract(self, *, player: Agent, designer: Agent) -> tuple[str, bool]:
        """
        Ask the LLM to confirm agreement to the contract with automatic retries.
        """
        base_prompt = (
            self.base_game.prompt
            + "\n"
            + self.contract_confirmation_prompt.format(
                contract_description=self._contract_description(
                    self.contracts[designer.name]
                )
            )
        )
        response, agreement = player.chat_with_retries(
            base_prompt=base_prompt,
            parse_func=self._parse_agreement,
        )
        return response, agreement

    def _parse_contract(self, response: str) -> list[int]:
        """
        Parse the contract design from the response.
        Expecting a Python dictionary in string format.
        """
        matches = re.findall(r"\{.*?\}", response, re.DOTALL)
        if not matches:
            raise ValueError(f"No JSON object found in the response {response!r}")
        json_str = matches[-1]

        try:
            json_obj = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e

        n = self.base_game.num_actions
        got_keys = set(json_obj.keys())
        missing = set(f"A{i}" for i in range(n)) - got_keys
        extra = got_keys - set(f"A{i}" for i in range(n))
        if extra:
            raise ValueError(f"Action key mismatch. Extra: {sorted(extra)}")
        if missing:
            raise ValueError(f"Action key mismatch. Missing: {sorted(missing)}")

        contract = [0] * n
        for k, v in json_obj.items():
            if not isinstance(v, int):
                raise ValueError(f"Value for {k} must be an integer, got {v!r}")
            idx = int(k[1:])  # strip the leading 'A'
            contract[idx] = v
        return contract

    def _parse_agreement(self, response: str) -> bool:
        """
        Parse the agreement to the contract from the response.
        Expecting a JSON object in string format.
        """
        matches = re.findall(r"\{.*?\}", response, re.DOTALL)
        if not matches:
            raise ValueError(f"No JSON object found in the response {response!r}")
        json_str = matches[-1]

        try:
            json_obj = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e

        if "sign" not in json_obj:
            raise ValueError(f"Missing 'sign' key in the response {response!r}")
        sign = json_obj["sign"]
        if not isinstance(sign, bool):
            raise ValueError(f"'sign' value must be a boolean, got {sign!r}")
        return sign

    def _contract_description(self, contract: list[int]) -> str:
        """Format the prompt for the contract agent.

        Args:
            contract (dict[int]): The contract with index representing the action
                and value representing the payoff adjustment.
        """
        lines = []
        for idx, payoff in enumerate(contract):
            if payoff > 0:
                lines.append(
                    f"- If you choose A{idx}, you receive a total of {payoff} from each other player."
                )
            elif payoff < 0:
                lines.append(
                    f"- If you choose A{idx}, you pay a total of {-payoff} to each other player."
                )
            else:
                lines.append(f"- If you choose A{idx}, there is no extra payoff.")
        return "\n".join(lines)

    def run_tournament(self, agents: Sequence[Agent]) -> PopulationPayoffs:
        contract_design = {}
        for agent in agents:
            response, contract = self._design_contract(agent)
            self.contracts[agent.name] = contract
            contract_design[agent.label] = {
                "response": response,
                "contract": contract,
            }
        LOGGER.log_record(record=contract_design, file_name="contract_design.json")
        return super().run_tournament(agents)

    def _play_matchup(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> None:
        """Have each designer propose a contract and play the base game."""

        history = []
        for designer in players:
            record = {
                "designer": designer.name,
                "agreements": {},
            }

            agreement_results = run_tasks(
                players,
                lambda ply: (
                    ply.label,
                    self._agree_to_contract(player=ply, designer=designer),
                ),
                max_workers=(
                    min(self.matchup_workers, len(players))
                    if getattr(self, "matchup_workers", 1) > 1
                    else 1
                ),
            )

            all_agree = True
            for label, (response, agree) in agreement_results:
                record["agreements"][label] = {
                    "response": response,
                    "agree": agree,
                }
                if not agree:
                    all_agree = False
            record["all_agree"] = all_agree

            if all_agree:
                contract_prompt = self.contract_mechanism_prompt.format(
                    contract_description=self._contract_description(
                        self.contracts[designer.name]
                    )
                )
                additional_info = [contract_prompt] * len(players)
            else:
                additional_info = ["None."] * len(players)

            moves = self.base_game.play(
                additional_info=additional_info,
                players=players,
            )
            payoffs.add_profile(moves)
            record["moves"] = [move.to_dict() for move in moves]
            history.append(record)

        LOGGER.log_record(record=history, file_name=self.record_file)
