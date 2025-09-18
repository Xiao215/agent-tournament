from __future__ import annotations

import textwrap
from typing import Callable, Mapping, Sequence

from src.agents.agent_manager import Agent
from src.games.base import Action, Game, Move


class PublicGoodsAction(Action):
    """Possible actions in the Public Goods game"""

    CONTRIBUTE = "C"
    FREE_RIDE = "F"


class PublicGoods(Game):
    """
    Public Goods environment in which N agents each decide to either
    contribute their full endowment or free-ride. After all choices are made,
    total contributions are multiplied by a factor and split equally.
    """

    def __init__(
        self,
        num_players: int,
        multiplier: float,
        *,
        parallel_players: bool = False,
    ) -> None:

        if num_players <= 1:
            raise ValueError("Public Goods must have at least 2 players.")
        if not (1.0 <= multiplier <= num_players):
            raise ValueError("Multiplier should be between 1 and num_players.")

        self.endowment = 1  # value of endowment does not matter
        self.multiplier = multiplier
        self.num_players = num_players
        self.parallel_players = parallel_players

        self.prompt_template = textwrap.dedent(
            """
        You are a player named {{player_name}} taking part in an N-player public goods game.
        Each player receives an endowment and simultaneously chooses whether to contribute it to a common pool.
        The pool is multiplied and shared equally among all players.

        Game rules:
        1. Every player starts with {endowment} dollar(s).
        2. Each player simultaneously chooses one action.
        3. Total contributions are multiplied by {multiplier} and divided equally among the group.
        4. Any money not contributed remains with the player.

        Actions (choose exactly one per round):
        - {contribute_tok} — Contribute your endowment to the pool.
        - {free_ride_tok} — Keep your endowment.
        """
        )

        super().__init__(
            prompt=self.prompt_template.format(
                endowment=self.endowment,
                multiplier=self.multiplier,
                num_players=num_players,
                contribute_tok=PublicGoodsAction.CONTRIBUTE.to_token(),
                free_ride_tok=PublicGoodsAction.FREE_RIDE.to_token(),
            ),
            num_players=num_players,
            num_actions=len(PublicGoodsAction),
        )

    def play(
        self,
        additional_info: list[str] | str,
        players: Sequence[Agent],
        action_map: Callable = lambda x: x,
    ) -> list[Move]:
        """
        Runs the Public Goods game: collects all actions, computes payoffs,
        and returns a list of Moves with each agent's action and points earned.
        """
        assert (
            len(players) == self.num_players
        ), f"Expected {self.num_players} agents, got {len(players)}."

        if isinstance(additional_info, str):
            additional_info = [additional_info] * self.num_players

        results = self._collect_actions(
            players,
            additional_info,
            parallel=self.parallel_players,
        )
        action_indices = {label: action_idx for label, action_idx, _ in results}
        responses = {label: resp for label, _, resp in results}
        labels_to_names = {
            label: players[idx].name for idx, (label, _, _) in enumerate(results)
        }

        mapped_indices = action_map(action_indices)
        final_actions: dict[str, PublicGoodsAction] = {
            lbl: PublicGoodsAction.from_index(action)
            for lbl, action in mapped_indices.items()
        }

        share = self._calculate_share(final_actions)

        moves = []
        for label, action in final_actions.items():
            name = labels_to_names[label]
            moves.append(
                Move(
                    name=name,
                    label=label,
                    action=action,
                    points=(
                        share
                        if action == PublicGoodsAction.CONTRIBUTE
                        else self.endowment + share
                    ),
                    response=responses[label],
                )
            )
        return moves

    def _calculate_share(self, actions: Mapping[str, PublicGoodsAction]) -> float:
        """
        Calculate the payoff for each agent based on their contributions.
        """

        contribution_count = sum(
            1 for v in actions.values() if v == PublicGoodsAction.CONTRIBUTE
        )

        return contribution_count * self.endowment * self.multiplier / self.num_players
