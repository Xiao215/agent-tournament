from enum import StrEnum

class Action(StrEnum):
    COOPERATE = "C"
    DEFECT = "D"

    def flip_action(self):
        return (
            Action.DEFECT
            if self == Action.COOPERATE
            else Action.COOPERATE
        )