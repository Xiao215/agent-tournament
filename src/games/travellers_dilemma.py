from __future__ import annotations

import textwrap
from enum import Enum
from typing import Callable, Iterable, Mapping, Sequence

from src.agents.agent_manager import Agent
from src.games.base import Action, Game, Move


def build_travellers_action(claims: Iterable[int]) -> type[Action]:
    """Create an Action enum for the given claim schedule."""
    claims = tuple(claims)
    if not claims:
        raise ValueError("claims must be a non-empty tuple.")
    members = {f"A{i}": int(claim) for i, claim in enumerate(claims)}
    return Enum("TravellersDilemmaAction", members, type=Action) # type: ignore[misc]


class TravellersDilemma(Game):
    """
    Traveler's Dilemma for two players with a configurable action set.

    Actions represent claims (integers). When claims differ, both players
    receive the lower claim, with a bonus added to the lower claimant and a
    penalty subtracted from the higher claimant.
    """

    def __init__(
        self,
        *,
        min_claim: int,
        num_actions: int,
        claim_spacing: int,
        bonus: float,
        parallel_players: bool = False,
    ) -> None:
        if num_actions < 2:
            raise ValueError("Travellers Dilemma requires at least 2 actions.")
        if claim_spacing <= 0:
            raise ValueError("claim_spacing must be a positive integer.")

        min_claim = int(min_claim)
        claim_spacing = int(claim_spacing)
        self.claims = tuple(min_claim + i * claim_spacing for i in range(num_actions))
        self.bonus = float(bonus)
        self.parallel_players = parallel_players

        self.action_cls = build_travellers_action(self.claims)

        actions_block = "\n".join(
            f"- {act.to_token()} — claim {act.value}" for act in self.action_cls
        )

        payoff_description = textwrap.dedent(
            f"""
        Payoff rule:
        - If both choose the same claim X: both receive X points.
        - If claims differ (X != Y): let M = min(X, Y).
          • The lower claimant receives M + {self.bonus}.
          • The higher claimant receives M - {self.bonus}.
        """
        ).strip()

        prompt_template = textwrap.dedent(
            """
        You are a player named {{player_name}} who plays in a decision-making game.
        Your objective is to maximise your score in the game described below.

        Rule:
        1. You and the opponent each choose a probability for each action, simultaneously.
        2. After both commitments are locked in, the final action will be drawn from the probability distributions.
        3. Both players receive the points specified by the rule below.

        Actions (each corresponds to a claim amount):
        {actions_block}

        {payoff_description}
        """
        )

        super().__init__(
            prompt=prompt_template.format(actions_block=actions_block, payoff_description=payoff_description),
            num_players=2,
            num_actions=len(self.claims),
        )

        # Override mixed-strategy instruction to reflect multi-action correctly
        self.default_output_instruction = textwrap.dedent(
            """
        Instruction:
        - Choose a probability distribution over ALL actions each round.
        - Output must contain a valid JSON object at the end.
        - Keys must be the action names exactly as given.
        - Values must be integers between 0 and 100.
        - All values must sum to exactly 100.

        Format requirement:
        Return exactly one JSON object, for example:
        {"A0": <INT>, "A1": <INT>, ...}
        """
        ).strip()

    def play(
        self,
        additional_info: list[str] | str,
        players: Sequence[Agent],
        action_map: Callable = lambda x: x,
    ) -> list[Move]:
        assert len(players) == 2
        player1, player2 = players

        if isinstance(additional_info, str):
            additional_info = [additional_info] * 2

        results = self._collect_actions(
            players,
            additional_info,
            parallel=self.parallel_players,
        )
        action_indices = {label: action_idx for label, action_idx, _ in results}
        responses = {label: resp for label, _, resp in results}

        mapped_indices = action_map(action_indices)

        # Map indices to actions (to preserve consistency with Move.action type)
        final_actions: dict[str, Action] = {
            lbl: self.action_cls.from_index(idx)
            for lbl, idx in mapped_indices.items()
        }

        # Compute payoffs from claims
        label1 = player1.label
        label2 = player2.label
        c1 = final_actions[label1].value
        c2 = final_actions[label2].value
        if c1 == c2:
            pts1 = pts2 = float(c1)
        else:
            m = float(min(c1, c2))
            if c1 < c2:
                pts1 = m + self.bonus
                pts2 = m - self.bonus
            else:
                pts1 = m - self.bonus
                pts2 = m + self.bonus

        return [
            Move(
                name=player1.name,
                label=label1,
                action=final_actions[label1],
                points=pts1,
                response=responses[label1],
            ),
            Move(
                name=player2.name,
                label=label2,
                action=final_actions[label2],
                points=pts2,
                response=responses[label2],
            ),
        ]

    @classmethod
    def parse_raw_payoff_matrix(
        cls,
        raw: Mapping[str, Sequence[float]],
        *,
        num_actions: int,
    ) -> dict[tuple[int, int], tuple[float, float]]:
        """
        Optional helper if a full payoff matrix is provided.

        Accepted key formats (i, j are action indices in [0, num_actions)):
        - "i,j" (e.g., "0,1")
        - "Ai,Aj" (e.g., "A0,A1")

        Values are two-element lists [p1, p2].
        """
        payoffs: dict[tuple[int, int], tuple[float, float]] = {}
        for key, val in raw.items():
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                raise ValueError(f"Invalid payoff for {key!r}: expected [p1, p2]")
            p1, p2 = float(val[0]), float(val[1])

            key = key.strip()
            if "," not in key:
                raise ValueError(
                    f"Invalid key {key!r}; expected 'i,j' or 'Ai,Aj' format."
                )
            a, b = [s.strip() for s in key.split(",", 1)]
            if a.startswith("A"):
                ai = int(a[1:])
                bi = int(b[1:])
            else:
                ai = int(a)
                bi = int(b)
            if not (0 <= ai < num_actions and 0 <= bi < num_actions):
                raise ValueError(
                    f"Action indices out of bounds in key {key!r}: got {(ai, bi)} with num_actions={num_actions}"
                )
            payoffs[(ai, bi)] = (p1, p2)
        return payoffs
