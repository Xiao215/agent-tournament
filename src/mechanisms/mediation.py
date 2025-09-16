import json
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Sequence

from src.agents.agent_manager import Agent
from src.evolution.population_payoffs import PopulationPayoffs
from src.games.base import Game
from src.logger_manager import LOGGER
from src.mechanisms.base import Mechanism


class Mediation(Mechanism):
    """
    Disarmament mechanism that allows for multiple rounds of the same game.
    """

    def __init__(
        self,
        base_game: Game,
    ) -> None:
        super().__init__(base_game)
        self.mediators: dict[str, dict[int, int]] = {}
        self.mediator_design_prompt = textwrap.dedent(
            """
        Instruction:
        You are tasked with designing a **mediator agent** for this game.

        - A mediator is an agent that suggests actions to players.
        - Each player may choose to **delegate** their move to the mediator or act independently.
        - The mediator observes number of players delegating to the mediator and then recommends the **same action** to all delegating players.
        - The mediator's objective is to maximize the payoff of the players who delegate to the mediator.

        Output Format:
        Return exactly **one valid Python dictionary** in a single line:
        {{"1": <Action>, ..., "{num_players}": <Action>}} where <Action> is a string like "A0", "A1" ...

        - Keys: the number of players delegating (from 1 to {num_players}).
        - Values: the action the mediator will recommend (e.g., "A0", "A1", ...).
        - Ensure the dictionary is syntactically valid in Python.
        """
        )

        self.game_prompt = textwrap.dedent(
            """
        Additional Information:
        On top of the original game instructions, you have the option to delegate your move to a mediator agent.
        If you choose to delegate, the mediator will play an action for you based on how many players have delegated to it.
        You can also choose to act independently.

        Here is what the mediator would do for the players that delegate to it:
        {mediator_description}

        Consider A{additional_action_id} as an addtional action "Delegate to Mediator". Your final mixed strategy should include probability for all actions A0, A1, ..., A{additional_action_id}.
        """
        )

    def _design_mediator(
        self, player: Agent, *, max_retries: int = 5
    ) -> tuple[str, dict[int, int]]:
        """
        Design the mediator agent by prompting the designer.

        Returns:
            response (str): The raw response from the designer.
            mediator (dict[int, int]): A dictionary mapping number of delegating players to recommended action.
        """
        base_prompt = self.mediator_design_prompt.format(
            num_players=self.base_game.num_players,
        )
        response = ""
        error_reason = ""

        # include initial + retries
        for attempt in range(max_retries + 1):
            if attempt == 0:
                prompt = base_prompt
            else:
                prompt = self._build_retry_prompt(base_prompt, response, error_reason)

            response = self.base_game.prompt_player(player, output_instruction=prompt)
            try:
                return response, self._parse_mediator(response)
            except ValueError as e:
                error_reason = str(e)
                print(
                    f"Attempt {attempt + 1} of {player.name} to design mediator failed: "
                    f"{error_reason} from response {response!r}"
                )
        raise ValueError(
            f"Failed to design mediator with {player.name} after {1 + max_retries} attempts. "
            f"Last error: {error_reason}. Last response: {response!r}"
        )

    def _parse_mediator(self, response: str) -> dict[int, int]:
        """
        Parse the mediator design from the response.
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

        mediator = {}
        for k, v in json_obj.items():
            k = int(k)
            if k < 1 or k > self.base_game.num_players:
                raise ValueError(
                    f"Invalid player number {k} for the pair {k}: {v}, "
                    f"must be between 1 and {self.base_game.num_players}."
                )
            if not 0 <= int(v[1:]) < self.base_game.num_actions:
                raise ValueError(
                    f"Invalid action {v} for the pair {k}: {v}, "
                    f"must be one of {[f'A{a}' for a in range(self.base_game.num_actions)]}."
                )
            mediator[k] = int(v[1:])
        if len(mediator) != self.base_game.num_players:
            raise ValueError(
                "There are missing cases in the mediator design, "
                f"you need to have cases for all number of players "
                f"from 1 to {self.base_game.num_players}."
            )
        return mediator

    def _mediator_description(self, mediator: dict[int, int]) -> str:
        """Format the prompt for the mediator agent."""
        lines = []
        for num_delegating, action in mediator.items():
            lines.append(
                f"\t• If {num_delegating} player(s) delegate to the mediator, "
                f"it will recommend action A{action}."
            )
        return "\n".join(lines)

    def run_tournament(self, agents: Sequence[Agent]) -> PopulationPayoffs:
        mediator_design = {}
        for agent in agents:
            if agent.name not in self.mediators:
                response, mediator = self._design_mediator(agent)
                self.mediators[agent.name] = mediator
                mediator_design[agent.name] = {
                    "response": response,
                    "mediator": mediator,
                }
        LOGGER.log_record(record=mediator_design, file_name="mediator_design.json")
        return super().run_tournament(agents)

    def _play_matchup(
        self, players: Sequence[Agent], payoffs: PopulationPayoffs
    ) -> None:
        history = []

        def play_for_mediator(player: Agent) -> tuple[str, list[dict]]:
            if player.name not in self.mediators:
                raise ValueError(f"Mediator for player {player.name} not found.")
            mediator = self.mediators[player.name]
            mediator_description = self._mediator_description(mediator)
            additional_info = self.game_prompt.format(
                mediator_description=mediator_description,
                additional_action_id=self.base_game.num_actions,
            )
            moves = self.base_game.play(
                players=players,
                additional_info=additional_info,
                action_map=self.mediator_mapping(mediator),
            )
            payoffs.add_profile(moves)
            return player.name, [move.to_dict() for move in moves]

        if self.matchup_workers <= 1 or len(players) <= 1:
            for player in players:
                mediator_name, move_dicts = play_for_mediator(player)
                history.append({"mediator": mediator_name, "moves": move_dicts})
        else:
            with ThreadPoolExecutor(max_workers=min(self.matchup_workers, len(players))) as ex:
                futures = {
                    ex.submit(play_for_mediator, player): player for player in players
                }
                for fut in futures:
                    mediator_name, move_dicts = fut.result()
                    history.append({"mediator": mediator_name, "moves": move_dicts})
        LOGGER.log_record(record=history, file_name=self.record_file)

    def mediator_mapping(self, mediator: dict[int, int]) -> Callable:
        """
        Given the original actions and the mediator design, return the final actions
        after applying the mediator's recommendations.
        """

        def apply_mediation(player_action_map: dict[str, int]) -> dict[str, int]:
            actions = {}
            num_delegating = sum(
                a == self.base_game.num_actions for a in player_action_map.values()
            )
            if num_delegating == 0:
                return player_action_map
            recommended_action = mediator[num_delegating]
            for player_name, action_idx in player_action_map.items():
                if action_idx == self.base_game.num_actions:
                    actions[player_name] = recommended_action
                else:
                    actions[player_name] = action_idx
            return actions

        return apply_mediation
