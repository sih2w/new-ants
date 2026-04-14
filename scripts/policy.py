from typing import TypedDict, List, TypeAlias
import numpy as np
from numpy.random import Generator
from scripts.env import EnvState, AGENT_ACTIONS
from scripts.vector import Vector2


DISCOUNT_FACTOR = 0.90
LEARNING_RATE = 0.10


QValues: TypeAlias = List[float]


PolicyGrid: TypeAlias = List[List[QValues]]


class PolicyLookup(TypedDict):
    HasFood: List[List[QValues]] # Row -> Column -> Q-values
    NoFood: List[List[List[QValues]]] # Remaining Food -> Row -> Column -> Q-values


class PolicyFunctions:
    @staticmethod
    def QValues() -> QValues:
        return [0.00] * len(AGENT_ACTIONS)

    @staticmethod
    def PolicyGrid(size: Vector2) -> PolicyGrid:
        return [[PolicyFunctions.QValues() for _ in range(size["Y"])] for _ in range(size["X"])]

    @staticmethod
    def PolicyLookup(size: Vector2, food_count: int) -> PolicyLookup:
        # If an agent is carrying food it will access HasFood. HasFood is a single grid of policies, where
        # each square contains QValues. If an agent does not have food, it will then access NoFood. NoFood is a list
        # of gridded policies, where the number of gridded policies corresponds to the initial number of food. For example,
        # if there is 5 food total and 3 food remaining, the agent will access the third grid of policies.
        return {
            "HasFood": PolicyFunctions.PolicyGrid(size),
            "NoFood": [PolicyFunctions.PolicyGrid(size) for _ in range(food_count + 1)],
        }

    @staticmethod
    def GetPolicyGrid(lookup: PolicyLookup, index: int, state: EnvState) -> PolicyGrid:
        if state["CarryingFood"][index]:
            return lookup["HasFood"]
        else:
            return lookup["NoFood"][state["FoodDeposited"]]

    @staticmethod
    def UpdatePolicy(
            lookup: PolicyLookup,
            agent_index: int,
            old_state: EnvState,
            new_state: EnvState,
            action: int,
            reward: float,
    ) -> None:
        old_policy_grid = PolicyFunctions.GetPolicyGrid(lookup, agent_index, old_state)
        new_policy_grid = PolicyFunctions.GetPolicyGrid(lookup, agent_index, new_state)

        old_location = old_state["AgentLocations"][agent_index]
        new_location = new_state["AgentLocations"][agent_index]

        old_q_values = old_policy_grid[old_location["X"]][old_location["Y"]]
        new_q_values = new_policy_grid[new_location["X"]][new_location["Y"]]

        predict = old_q_values[action]
        target = reward + DISCOUNT_FACTOR * max(new_q_values)

        old_q_values[action] += LEARNING_RATE * (target - predict)

    @staticmethod
    def GetAction(
            lookup: PolicyLookup,
            agent_index: int,
            generator: Generator,
            state: EnvState,
            epsilon: float
    ) -> int:
        if generator.random() > epsilon:
            location = state["AgentLocations"][agent_index]
            policy_grid = PolicyFunctions.GetPolicyGrid(lookup, agent_index, state)
            return int(np.argmax(policy_grid[location["X"]][location["Y"]]))
        else:
            return int(generator.integers(low=0, high=len(AGENT_ACTIONS)))
