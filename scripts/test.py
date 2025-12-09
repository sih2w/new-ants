from typing import List, Any
from scripts.env import EnvFunctions, Env, EnvParams, Agent, EnvState
from scripts.datastore import DataStoreFunctions
from scripts.event import EventFunctions
from scripts.episode import Episode, EpisodeFunctions
from scripts.policy import PolicyFunctions, PolicyLookup
from scripts.vector import Vector2


class EnvTest:
    Actions: List[int] = []
    Rewards: List[int] = []
    Epsilon: float = 1
    Lookups: List[PolicyLookup]
    Episodes: List[Episode]
    DecayRate: float
    CurrentEpisode: Episode
    Env: Env

    @staticmethod
    def BeforeDeposited(food_index: int) -> bool:
        for index, food in enumerate(env["Food"]):
            if index < food_index and food["Status"] != "Deposited":
                return False
        return True

    @staticmethod
    def AfterDeposited(food_index: int) -> bool:
        for index, food in enumerate(env["Food"]):
            if index > food_index and food["Status"] != "Deposited":
                return False
        return True

    @staticmethod
    def UpdateAgent1(agent: Agent, action: int):
        success = EnvFunctions.TryMoveAgent(EnvTest.Env, agent, action)
        if not success:
            return -1000

        food = EnvFunctions.OnDroppedFood(EnvTest.Env, agent["Location"])
        if food and EnvFunctions.CanPickup(agent, food):
            # Only pickup if all prior food has been deposited.
            if EnvTest.BeforeDeposited(EnvTest.Env["Food"].index(food)):
                EnvFunctions.GiveFood(agent, food)
                return 10

        nest = EnvFunctions.OnNest(EnvTest.Env, agent["Location"])
        if nest:
            for food in agent["Food"]:
                if EnvFunctions.CanDeposit(EnvTest.Env, agent, food):
                    EnvFunctions.Deposit(EnvTest.Env, agent, food)
                    return 10
        return -1

    @staticmethod
    def UpdateAgent2(agent: Agent, action: int):
        success = EnvFunctions.TryMoveAgent(EnvTest.Env, agent, action)
        if not success:
            return -1000

        food = EnvFunctions.OnDroppedFood(EnvTest.Env, agent["Location"])
        if food and EnvFunctions.CanPickup(agent, food):
            # Only pickup if all food after has been deposited.
            if EnvTest.AfterDeposited(EnvTest.Env["Food"].index(food)):
                EnvFunctions.GiveFood(agent, food)
                return 10

        nest = EnvFunctions.OnNest(EnvTest.Env, agent["Location"])
        if nest:
            for food in agent["Food"]:
                if EnvFunctions.CanDeposit(EnvTest.Env, agent, food):
                    EnvFunctions.Deposit(EnvTest.Env, agent, food)
                    return 10
        return -1

    @staticmethod
    def UpdateAgent3(agent: Agent, action: int):
        success = EnvFunctions.TryMoveAgent(EnvTest.Env, agent, action)
        if not success:
            return -1000

        food = EnvFunctions.OnDroppedFood(EnvTest.Env, agent["Location"])
        if food and EnvFunctions.CanPickup(agent, food):
            EnvFunctions.GiveFood(agent, food)
            return 10

        nest = EnvFunctions.OnNest(EnvTest.Env, agent["Location"])
        if nest:
            for food in agent["Food"]:
                if EnvFunctions.CanDeposit(EnvTest.Env, agent, food):
                    EnvFunctions.Deposit(EnvTest.Env, agent, food)
                    return 10
        return -1

    @staticmethod
    def UpdateAgent(agent: Agent, index: int, action: int):
        # if index == 1:
        #     return EnvTest.UpdateAgent1(agent, action)
        # elif index == 2:
        #     return EnvTest.UpdateAgent2(agent, action)
        return EnvTest.UpdateAgent3(agent, action)

    @staticmethod
    def QAction(agent_index: int, state: EnvState):
        return PolicyFunctions.GetAction(
            lookup=EnvTest.Lookups[agent_index],
            agent_index=agent_index,
            generator=EnvTest.Env["Generator"],
            state=state,
            epsilon=EnvTest.Epsilon
        )

    @staticmethod
    def OnTrainingStepStarted(message: Any):
        # Clear the last step's actions and rewards.
        EnvTest.Actions.clear()
        EnvTest.Rewards.clear()

        # Update each agent using Q-learning.
        for index, agent in enumerate(EnvTest.Env["Agents"]):
            EnvTest.Actions.append(EnvTest.QAction(index, message["State"]))
            EnvTest.Rewards.append(EnvTest.UpdateAgent(agent, index, EnvTest.Actions[index]))
            agent["LastAction"] = EnvTest.Actions[index]

    @staticmethod
    def OnTrainingStepEnded(message: Any):
        total_rewards, count = 0, 1
        # Update each agent's policy with the chosen action and resulting rewards.
        for index, agent in enumerate(EnvTest.Env["Agents"]):
            total_rewards += EnvTest.Rewards[index]
            PolicyFunctions.UpdatePolicy(
                lookup=lookups[index],
                agent_index=index,
                old_state=message["OldState"],
                new_state=message["NewState"],
                action=EnvTest.Actions[index],
                reward=EnvTest.Rewards[index],
            )

        # Add the average reward to the current episode and reduce epsilon.
        EnvTest.CurrentEpisode["AverageRewards"].append(total_rewards / count)
        EnvTest.Epsilon -= EnvTest.DecayRate

    @staticmethod
    def OnTestingStepStarted(message: Any):
        EnvTest.Epsilon = 0 # Set to zero to get the most optimal action.

        for index, agent in enumerate(EnvTest.Env["Agents"]):
            agent["LastAction"] = EnvTest.QAction(index, message["State"])
            EnvTest.UpdateAgent(agent, index, agent["LastAction"])

    @staticmethod
    def OnRendered(message: Any):
        def callback(agent_index: int, location: Vector2):
            message["State"]["AgentLocations"][agent_index] = location
            return EnvTest.QAction(agent_index, message["State"])
        EnvFunctions.DrawArrows(EnvTest.Env, callback, message["Surface"])

    @staticmethod
    def OnEpisodeStarted(message: Any):
        pass

    @staticmethod
    def OnEpisodeEnded(message: Any):
        # Add the current episode to the episode list and then create a new episode.
        EnvTest.Episodes.append(EnvTest.CurrentEpisode)
        EnvTest.CurrentEpisode = EpisodeFunctions.Episode()

    @staticmethod
    def OnProximityDetected(message: Any):
        index1 = EnvTest.Env["Agents"].index(message["Agent1"])
        index2 = EnvTest.Env["Agents"].index(message["Agent2"])
        state = EnvFunctions.GetState(EnvTest.Env)

        if state["CarryingFood"][index1] == state["CarryingFood"][index2]:
            state["AgentLocations"] = [message["Agent1"]["Location"]] * len(EnvTest.Env["Agents"])
            policy1 = PolicyFunctions.GetPolicy(
                lookup=EnvTest.Lookups[index1],
                index=index1,
                state=state,
            )

            policy2 = PolicyFunctions.GetPolicy(
                lookup=EnvTest.Lookups[index2],
                index=index2,
                state=state,
            )

            for index, value in enumerate(policy1["QValues"]):
                policy1["QValues"][index] = (value + policy2["QValues"][index]) / 2
                policy2["QValues"][index] = policy1["QValues"][index]

if __name__ == "__main__":
    params: EnvParams = {
        "AgentCount": 2,
        "FoodCount": 10,
        "ObstacleCount": 10,
        "NestCount": 1,
        "GridSize": {"X": 15, "Y": 15},
        "Seed": 9,
        "MaxSteps": 10_000,
        "EpisodeCount": 1000,
        "ProximityRadius": 0.00,
    }

    lookups, episodes = DataStoreFunctions.Load(params)
    env: Env = EnvFunctions.Env(params)

    # Config the custom functions.
    EnvTest.Lookups = lookups
    EnvTest.Episodes = episodes
    EnvTest.DecayRate = 1 / params["EpisodeCount"]
    EnvTest.Env = env
    EnvTest.CurrentEpisode = EpisodeFunctions.Episode()

    # Initialize pygame and the env.
    EnvFunctions.Init(env)

    if len(episodes) == 0:
        # Connect the training events and start training.
        EventFunctions.Connect(env["StepStarted"], EnvTest.OnTrainingStepStarted)
        EventFunctions.Connect(env["StepEnded"], EnvTest.OnTrainingStepEnded)
        EventFunctions.Connect(env["EpisodeStarted"], EnvTest.OnEpisodeStarted)
        EventFunctions.Connect(env["EpisodeEnded"], EnvTest.OnEpisodeEnded)
        # EventFunctions.Connect(env["ProximityDetected"], EnvTest.OnProximityDetected)

        EnvFunctions.RunTrain(env)

        # Disconnect the training events.
        EventFunctions.DisconnectAll(env["StepStarted"])
        EventFunctions.DisconnectAll(env["StepEnded"])
        EventFunctions.DisconnectAll(env["EpisodeStarted"])
        EventFunctions.DisconnectAll(env["EpisodeEnded"])

        # Save the training results.
        DataStoreFunctions.Save(params, lookups, episodes)

    # Plot the training results.
    EpisodeFunctions.PlotRewards(episodes)
    EpisodeFunctions.PlotSteps(episodes)

    # Draw decision arrows on render.
    # EventFunctions.Connect(env["Rendered"], EnvConfig.OnRendered)

    # Connect the testing events and view result of training.
    EventFunctions.Connect(env["StepStarted"], EnvTest.OnTestingStepStarted)
    EnvFunctions.RunTest(env)