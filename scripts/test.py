from copy import copy
from typing import Any, Tuple
import pygame
from scripts.env import EnvFunctions, Env, EnvParams, Agent, EnvState
from scripts.episode import Episode, EpisodeFunctions
from scripts.policy import PolicyFunctions, PolicyLookup
from scripts.vector import Vector2
from typing import List, TypedDict


class EnvTest(TypedDict):
    AgentArrowIndex: int
    Actions: List[int]
    Rewards: List[int]
    Epsilon: float
    Lookups: List[PolicyLookup]
    Episodes: List[Episode]
    DecayRate: float
    CurrentEpisode: Episode
    agent_1FoodIndex: int
    agent_2FoodIndex : int
    FoodPriorities: Tuple[Tuple[int]]
    Env: Env


class EnvTestFunctions:
    @staticmethod
    def EnvTest(params: EnvParams) -> EnvTest:
        return {
            "AgentArrowIndex": 0,
            "Actions": [],
            "Rewards": [],
            "Epsilon": 1,
            "Lookups": [],
            "Episodes": [],
            "DecayRate": 1 / params["EpisodeCount"],
            "CurrentEpisode": EpisodeFunctions.Episode(),
            "Agent1FoodIndex": 0,
            "Agent2FoodIndex": 14,
            "FoodPriorities": (
                (0, 1, 2, 3, 4),
                (5, 6, 7, 8, 9),
                (10, 11, 12, 13, 14),
            ),
            "Env": EnvFunctions.Env(params)
        }

    @staticmethod
    def OnMaxStepReached(env: EnvTest):
        env["CurrentEpisode"]["TerminatedEarly"] = True
        for food in env["Env"]["Food"]:
            food["Status"] = "Deposited"

    @staticmethod
    def UpdateAgentOrdered(env: EnvTest, agent: Agent, action: int):
        success = EnvFunctions.TryMoveAgent(env["Env"], agent, action)
        if not success:
            return -1000

        food = EnvFunctions.OnDroppedFood(env["Env"], agent["Location"])
        if food and EnvFunctions.CanPickup(agent, food):
            EnvFunctions.GiveFood(agent, food)

            food_index = env["Env"]["Food"].index(food)
            agent_index = env["Env"]["Agents"].index(agent)

            if agent_index == 0:
                if food_index == env["Agent1FoodIndex"]:
                    env["Agent1FoodIndex"] += 1
                    return 100
            else:
                if food_index == env["Agent2FoodIndex"]:
                    env["Agent2FoodIndex"] -= 1
                    return 100
            return 10

        nest = EnvFunctions.OnNest(env["Env"], agent["Location"])
        if nest:
            for food in agent["Food"]:
                if EnvFunctions.CanDeposit(env["Env"], agent, food):
                    EnvFunctions.Deposit(env["Env"], agent, food)
                    return 10
        return -1

    @staticmethod
    def UpdateAgentDefault(env: EnvTest, agent: Agent, action: int):
        success = EnvFunctions.TryMoveAgent(env["Env"], agent, action)
        if not success:
            return -1000

        food = EnvFunctions.OnDroppedFood(env["Env"], agent["Location"])
        if food and EnvFunctions.CanPickup(agent, food):
            EnvFunctions.GiveFood(agent, food)
            return 10

        nest = EnvFunctions.OnNest(env["Env"], agent["Location"])
        if nest:
            for food in agent["Food"]:
                if EnvFunctions.CanDeposit(env["Env"], agent, food):
                    EnvFunctions.Deposit(env["Env"], agent, food)
                    return 10
        return -1

    @staticmethod
    def UpdateAgentWithPriority(env: EnvTest, agent: Agent, action: int):
        success = EnvFunctions.TryMoveAgent(env["Env"], agent, action)
        if not success:
            return -1000

        food = EnvFunctions.OnDroppedFood(env["Env"], agent["Location"])
        if food and EnvFunctions.CanPickup(agent, food):
            agent_index = env["Env"]["Agents"].index(agent)
            food_index = env["Env"]["Food"].index(food)

            EnvFunctions.GiveFood(agent, food)

            return 100 if food_index in env["FoodPriorities"][agent_index] else 10

        nest = EnvFunctions.OnNest(env["Env"], agent["Location"])
        if nest:
            for food in agent["Food"]:
                if EnvFunctions.CanDeposit(env["Env"], agent, food):
                    EnvFunctions.Deposit(env["Env"], agent, food)
                    return 10
        return -1

    @staticmethod
    def QAction(env: EnvTest, agent_index: int, state: EnvState):
        return PolicyFunctions.GetAction(
            lookup=env["Lookups"][agent_index],
            agent_index=agent_index,
            generator=env["Env"]["Generator"],
            state=state,
            epsilon=env["Epsilon"]
        )

    @staticmethod
    def OnTrainingStepEnded(env: EnvTest, message: Any):
        total_rewards = 0
        # Update each agent's policy with the chosen action and resulting rewards.
        for index, agent in enumerate(env["Env"]["Agents"]):
            total_rewards += env["Rewards"][index]
            PolicyFunctions.UpdatePolicy(
                lookup=env["Lookups"][index],
                agent_index=index,
                old_state=message["OldState"],
                new_state=message["NewState"],
                action=env["Actions"][index],
                reward=env["Rewards"][index],
            )

        env["CurrentEpisode"]["TotalRewards"] += total_rewards
        env["CurrentEpisode"]["Steps"] += 1
        env["Epsilon"] -= env["DecayRate"]

    @staticmethod
    def OnRendered(env: EnvTest, message: Any):
        grid_actions = []
        for x in range(env["Env"]["GridSize"]["X"]):
            for y in range(env["Env"]["GridSize"]["Y"]):
                location: Vector2 = {"X": x, "Y": y}
                message["State"]["AgentLocations"][env["AgentArrowIndex"]] = location
                action = EnvTestFunctions.QAction(env, env["AgentArrowIndex"], message["State"])
                grid_actions.append((location, action))
        EnvFunctions.DrawArrows(env["Env"], env["AgentArrowIndex"], grid_actions, message["Surface"])

    @staticmethod
    def OnTicked(env: EnvTest, message: Any):
        current = env["AgentArrowIndex"]
        if pygame.key.get_pressed()[pygame.K_LEFT]:
            env["AgentArrowIndex"] -= 1
        elif pygame.key.get_pressed()[pygame.K_RIGHT]:
            env["AgentArrowIndex"] += 1

        if current != env["AgentArrowIndex"]:
            current = min(max(env["AgentArrowIndex"], 0), len(env["Env"]["Agents"]) - 1)
            env["AgentArrowIndex"] = current
            EnvFunctions.RenderFrame(env["Env"])

    @staticmethod
    def OnEpisodeEnded(env: EnvTest, message: Any):
        # Add the current episode to the episode list and then create a new episode.
        env["Episodes"].append(env["CurrentEpisode"])
        env["CurrentEpisode"] = EpisodeFunctions.Episode()
        env["Agent1FoodIndex"] = 0
        env["Agent2FoodIndex"] = 14

    @staticmethod
    def OnReset(env: EnvTest, message: Any):
        if env["Env"]["StepCount"] > 0:
            print(env["Env"]["StepCount"])
            env["Env"]["Running"] = False

    @staticmethod
    def SameStatus(agent_1: Agent, agent_2: Agent):
        return len(agent_1["Food"]) == len(agent_2["Food"])

    @staticmethod
    def OnProximityDetectedNoExchange(env: EnvTest, message: Any):
        agent_1 = message["Agent1"]
        agent_2 = message["Agent2"]

        if EnvTestFunctions.SameStatus(agent_1, agent_2):
            env["CurrentEpisode"]["PotentialExchanges"] += 1

    @staticmethod
    def OnProximityDetectedAverage(env: EnvTest, message: Any):
        agent_1 = message["Agent1"]
        agent_2 = message["Agent2"]

        if EnvTestFunctions.SameStatus(agent_1, agent_2):
            env["CurrentEpisode"]["PotentialExchanges"] += 1
            env["CurrentEpisode"]["Exchanges"] += 1

            index_1 = env["Env"]["Agents"].index(agent_1)
            index_2 = env["Env"]["Agents"].index(agent_2)
            state = EnvFunctions.GetState(env["Env"])
            
            policy_grid_1 = PolicyFunctions.GetPolicyGrid(
                lookup=env["Lookups"][index_1],
                index=index_1,
                state=state,
            )

            policy_grid_2 = PolicyFunctions.GetPolicyGrid(
                lookup=env["Lookups"][index_2],
                index=index_2,
                state=state,
            )

            for x in range(env["Env"]["GridSize"]["X"]):
                for y in range(env["Env"]["GridSize"]["Y"]):
                    q_values_1 = policy_grid_1[x][y]
                    q_values_2 = policy_grid_2[x][y]
                    lists = [q_values_1, q_values_2]

                    average_q_values = [sum(q_values) / len(q_values) for q_values in zip(*lists)]
                    policy_grid_1[x][y] = copy(average_q_values)
                    policy_grid_2[x][y] = copy(average_q_values)

    @staticmethod
    def OnProximityDetectedFill(env: EnvTest, message: Any):
        agent_1 = message["Agent1"]
        agent_2 = message["Agent2"]

        if EnvTestFunctions.SameStatus(agent_1, agent_2):
            env["CurrentEpisode"]["PotentialExchanges"] += 1

            index_1 = env["Env"]["Agents"].index(agent_1)
            index_2 = env["Env"]["Agents"].index(agent_2)
            state = EnvFunctions.GetState(env["Env"])

            policy_grid_1 = PolicyFunctions.GetPolicyGrid(
                lookup=env["Lookups"][index_1],
                index=index_1,
                state=state,
            )

            policy_grid_2 = PolicyFunctions.GetPolicyGrid(
                lookup=env["Lookups"][index_2],
                index=index_2,
                state=state,
            )

            exchanged = False

            for x in range(env["Env"]["GridSize"]["X"]):
                for y in range(env["Env"]["GridSize"]["Y"]):
                    q_values_1 = policy_grid_1[x][y]
                    q_values_2 = policy_grid_2[x][y]

                    if all(v == 0 for v in q_values_1) and not all(v == 0 for v in q_values_2):
                        policy_grid_1[x][y] = copy(q_values_2)
                        exchanged = True

                    if all(v == 0 for v in q_values_2) and not all(v == 0 for v in q_values_1):
                        policy_grid_2[x][y] = copy(q_values_1)
                        exchanged = True

            if exchanged:
                env["CurrentEpisode"]["Exchanges"] += 1