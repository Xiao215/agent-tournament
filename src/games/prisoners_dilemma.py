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

        {{instruction}}
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

        responses = {}
        actions = {}

        def play_one(player: Agent, info: str) -> tuple[str, int, str]:
            resp = self.prompt_player(player, info)
            probs = self._extract_mixed_strategy(player, resp, info)
            action_idx = self._choose_from_mix_strategy(probs)
            return player.name, action_idx, resp

        if self.parallel_players:
            with ThreadPoolExecutor(max_workers=self.num_players) as ex:
                futs = [ex.submit(play_one, p, info) for p, info in zip(players, additional_info)]
                for fut in futs:
                    name, action_idx, resp = fut.result()
                    actions[name] = action_idx
                    responses[name] = resp
        else:
            for player, info in zip(players, additional_info):
                name, action_idx, resp = play_one(player, info)
                actions[name] = action_idx
                responses[name] = resp

        actions = action_map(actions)
        actions = {
            name: PrisonersDilemmaAction.from_index(action)
            for name, action in actions.items()
        }

        pts1, pts2 = self.payoff_matrix[(actions[player1.name], actions[player2.name])]
        return [
            Move(
                name=player1.name,
                action=actions[player1.name],
                points=pts1,
                response=responses[player1.name],
            ),
            Move(
                name=player2.name,
                action=actions[player2.name],
                points=pts2,
                response=responses[player2.name],
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
