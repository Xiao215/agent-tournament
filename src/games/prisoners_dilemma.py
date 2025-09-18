import textwrap
from typing import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from src.agents.agent_manager import Agent
from src.games.base import Action, Game, Move


class PrisonersDilemmaAction(Action):
    """Possible actions in the Prisoner's Dilemma"""

    COOPERATE = "C"
    DEFECT = "D"


class PrisonersDilemma(Game):
    """
    Prisoner's Dilemma environment that allows for one rounds of interaction
    between two LLM agents.
    """

    def __init__(
        self,
        payoff_matrix: dict[str, list[float]],
        *,
        parallel_players: bool = False,
    ) -> None:
        self.payoff_matrix = self._parse_payoff_matrix(payoff_matrix)
        self.parallel_players = parallel_players

        self.action_tokens = [act.to_token() for act in PrisonersDilemmaAction]
        actions_block = "\n".join([f"- {act_tok}" for act_tok in self.action_tokens])
        self.prompt_template = textwrap.dedent(
            """
        You are a player named {{player_name}} who plays in a decision-making game.
        Your objective is to maximise your score in the game described below.

        Rule:
        1. You and the opponent each choose a probability for each action, simultaneously.
        2. After both commitments are locked in, the final action will be drawn from the probability distributions.
        3. Both players receive the points specified in the payoff table below.

        Actions:
        {actions_block}

        Payoff matrix:
        {payoff_description}
        """
        )

        super().__init__(
            prompt=self.prompt_template.format(
                actions_block=actions_block,
                payoff_description=self._payoff_description(),
            ),
            num_players=2,
            num_actions=len(PrisonersDilemmaAction),
        )

    def _payoff_description(self) -> str:
        lines = []
        for (a, b), (pts_a, pts_b) in self.payoff_matrix.items():
            lines.append(
                f"  • If you choose {a.to_token()} and opponent chooses {b.to_token()}: "
                f"you get {pts_a} points, opponent gets {pts_b} points."
            )
        return "\n".join(lines)

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

        responses: dict[str, str] = {}
        action_indices: dict[str, int] = {}

        def play_one(player: Agent, info: str) -> tuple[str, int, str]:
            resp, mix_probs = self.prompt_player_mix_probs(player, info)
            action_idx = self._choose_from_mix_strategy(mix_probs)
            return player.label, action_idx, resp

        if self.parallel_players:
            with ThreadPoolExecutor(max_workers=self.num_players) as ex:
                futs = [ex.submit(play_one, p, info) for p, info in zip(players, additional_info)]
                for fut in futs:
                    label, action_idx, resp = fut.result()
                    action_indices[label] = action_idx
                    responses[label] = resp
        else:
            for player, info in zip(players, additional_info):
                label, action_idx, resp = play_one(player, info)
                action_indices[label] = action_idx
                responses[label] = resp

        mapped_indices = action_map(action_indices)
        final_actions: dict[str, PrisonersDilemmaAction] = {
            lbl: PrisonersDilemmaAction.from_index(action)
            for lbl, action in mapped_indices.items()
        }

        label1 = player1.label
        label2 = player2.label
        pts1, pts2 = self.payoff_matrix[(final_actions[label1], final_actions[label2])]
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
    def _parse_payoff_matrix(
        cls,
        raw_payoff: dict[str, list[float]],
    ) -> dict[
        tuple[PrisonersDilemmaAction, PrisonersDilemmaAction], tuple[float, float]
    ]:
        """
        Convert a raw payoff matrix with string keys into typed action pairs.
        """
        payoffs = {}
        for key, (p1, p2) in raw_payoff.items():
            a1 = PrisonersDilemmaAction(key[0])
            a2 = PrisonersDilemmaAction(key[1])
            payoffs[(a1, a2)] = (p1, p2)
        return payoffs
