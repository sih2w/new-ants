from typing import TypedDict, List
import numpy as np
from numpy.random import Generator
from scripts.env import Agent, EnvState, AGENT_ACTIONS
from vector import Vector2


DISCOUNT_FACTOR = 0.90
LEARNING_RATE = 0.10


class Policy(TypedDict):
    QValues: List[float]


class PolicyLookup(TypedDict):
    HasFood: List[List[Policy]] # Row -> Column -> Policy
    NoFood: List[List[List[Policy]]] # Remaining Food -> Row -> Column -> Policy


class PolicyFunctions:
    @staticmethod
    def Policy() -> Policy:
        return {
            "QValues": [0.00] * len(AGENT_ACTIONS),
        }

    @staticmethod
    def GridOfPolicies(size: Vector2) -> List[List[Policy]]:
        return [[PolicyFunctions.Policy() for _ in range(size["Y"])] for _ in range(size["X"])]

    @staticmethod
    def PolicyLookup(size: Vector2, food_count: int) -> PolicyLookup:
        # If an agent is carrying food it will access HasFood. HasFood is a single grid of policies, where
        # each square contains QValues. If an agent does not have food, it will then access NoFood. NoFood is a list
        # of gridded policies, where the number of gridded policies corresponds to the initial number of food. For example,
        # if there is 5 food total and 3 food remaining, the agent will access the third grid of policies.
        return {
            "HasFood": PolicyFunctions.GridOfPolicies(size),
            "NoFood": [PolicyFunctions.GridOfPolicies(size) for _ in range(food_count + 1)],
        }

    @staticmethod
    def GetPolicy(lookup: PolicyLookup, index: int, state: EnvState) -> Policy:
        location = state["AgentLocations"][index]
        if state["CarryingFood"][index]:
            return lookup["HasFood"][location["X"]][location["Y"]]
        else:
            return lookup["NoFood"][state["FoodDeposited"]][location["X"]][location["Y"]]

    @staticmethod
    def UpdatePolicy(
            lookup: PolicyLookup,
            agent_index: int,
            old_state: EnvState,
            new_state: EnvState,
            action: int,
            reward: float,
    ) -> None:
        oldPolicy = PolicyFunctions.GetPolicy(lookup, agent_index, old_state)
        newPolicy = PolicyFunctions.GetPolicy(lookup, agent_index, new_state)

        predict = oldPolicy["QValues"][action]
        target = reward + DISCOUNT_FACTOR * max(newPolicy["QValues"])

        oldPolicy["QValues"][action] += LEARNING_RATE * (target - predict)
        return None

    @staticmethod
    def GetAction(
            lookup: PolicyLookup,
            agent_index: int,
            generator: Generator,
            state: EnvState,
            epsilon: float
    ) -> int:
        if generator.random() > epsilon:
            policy = PolicyFunctions.GetPolicy(lookup, agent_index, state)
            return int(np.argmax(policy["QValues"]))
        else:
            return generator.integers(low=0, high=len(AGENT_ACTIONS))
