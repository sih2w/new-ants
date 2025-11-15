from typing import List, Any
from scripts.env import EnvFunctions, Env, EnvParams, Agent, EnvState
from scripts.datastore import DataStoreFunctions
from scripts.event import EventFunctions
from scripts.episode import Episode, EpisodeFunctions
from scripts.policy import PolicyFunctions, PolicyLookup
from scripts.vector import Vector2


class EnvConfig:
    # These variables are shared among the class's static functions.
    Actions: List[int] = []
    Rewards: List[int] = []
    Epsilon: float = 1
    Lookups: List[PolicyLookup]
    Episodes: List[Episode]
    DecayRate: float
    CurrentEpisode: Episode
    Env: Env

    def __init__(
            self,
            lookup: List[PolicyLookup],
            episodes: List[Episode],
            episode_count: int,
            env: Env
    ):
        # Initialize custom function fields.
        EnvConfig.Lookups = lookup
        EnvConfig.Episodes = episodes
        EnvConfig.DecayRate = 1 / episode_count
        EnvConfig.Env = env
        EnvConfig.CurrentEpisode = EpisodeFunctions.Episode()

    @staticmethod
    def UpdateAgent(agent: Agent, action: int):
        success = EnvFunctions.TryMoveAgent(EnvConfig.Env, agent, action)
        if not success:
            return -1000

        food = EnvFunctions.OnDroppedFood(EnvConfig.Env, agent["Location"])
        if food and EnvFunctions.CanPickup(agent, food):
            EnvFunctions.GiveFood(agent, food)
            return 10

        nest = EnvFunctions.OnNest(EnvConfig.Env, agent["Location"])
        if nest:
            for food in agent["Food"]:
                if EnvFunctions.CanDeposit(EnvConfig.Env, agent, food):
                    EnvFunctions.Deposit(EnvConfig.Env, agent, food)
                    return 10
        return -1

    @staticmethod
    def QAction(agent_index: int, state: EnvState):
        return PolicyFunctions.GetAction(
            lookup=EnvConfig.Lookups[agent_index],
            agent_index=agent_index,
            generator=EnvConfig.Env["Generator"],
            state=state,
            epsilon=EnvConfig.Epsilon
        )

    @staticmethod
    def OnTrainingStepStarted(message: Any):
        # Clear the last step's actions and rewards.
        EnvConfig.Actions.clear()
        EnvConfig.Rewards.clear()

        # Update each agent using Q-learning.
        for index, agent in enumerate(EnvConfig.Env["Agents"]):
            EnvConfig.Actions.append(EnvConfig.QAction(index, message["State"]))
            EnvConfig.Rewards.append(EnvConfig.UpdateAgent(agent, EnvConfig.Actions[index]))
            agent["LastAction"] = EnvConfig.Actions[index]

    @staticmethod
    def OnTrainingStepEnded(message: Any):
        total_rewards, count = 0, 1
        # Update each agent's policy with the chosen action and resulting rewards.
        for index, agent in enumerate(EnvConfig.Env["Agents"]):
            total_rewards += EnvConfig.Rewards[index]
            PolicyFunctions.UpdatePolicy(
                lookup=lookups[index],
                agent_index=index,
                old_state=message["OldState"],
                new_state=message["NewState"],
                action=EnvConfig.Actions[index],
                reward=EnvConfig.Rewards[index],
            )

        # Add the average reward to the current episode and reduce epsilon.
        EnvConfig.CurrentEpisode["AverageRewards"].append(total_rewards / count)
        EnvConfig.Epsilon -= EnvConfig.DecayRate

    @staticmethod
    def OnTestingStepStarted(message: Any):
        EnvConfig.Epsilon = 0 # Set to zero to get the most optimal action.

        for index, agent in enumerate(EnvConfig.Env["Agents"]):
            agent["LastAction"] = EnvConfig.QAction(index, message["State"])
            EnvConfig.UpdateAgent(agent, agent["LastAction"])

    @staticmethod
    def OnRendered(message: Any):
        def callback(agent_index: int, location: Vector2):
            message["State"]["AgentLocations"][agent_index] = location
            return EnvConfig.QAction(agent_index, message["State"])
        EnvFunctions.DrawArrows(EnvConfig.Env, callback, message["Surface"])

    @staticmethod
    def OnEpisodeStarted(message: Any):
        pass

    @staticmethod
    def OnEpisodeEnded(message: Any):
        # Add the current episode to the episode list and then create a new episode.
        EnvConfig.Episodes.append(EnvConfig.CurrentEpisode)
        EnvConfig.CurrentEpisode = EpisodeFunctions.Episode()

    @staticmethod
    def OnProximityDetected(message: Any):
        # TODO: Exchange logic here.
        pass

if __name__ == "__main__":
    params: EnvParams = {
        "AgentCount": 1,
        "FoodCount": 30,
        "ObstacleCount": 10,
        "NestCount": 1,
        "GridSize": {"X": 10, "Y": 10},
        "Seed": 1,
        "MaxSteps": 10_000_000,
        "EpisodeCount": 10_000,
        "ProximityRadius": 1.00,
    }

    lookups, episodes = DataStoreFunctions.Load(params)
    env: Env = EnvFunctions.Env(params)

    # Config the custom functions.
    EnvConfig(lookups, episodes, params["EpisodeCount"], env)

    # Initialize pygame and the env.
    EnvFunctions.Init(env)

    if len(episodes) == 0:
        # Connect the training events and start training.
        EventFunctions.Connect(env["StepStarted"], EnvConfig.OnTrainingStepStarted)
        EventFunctions.Connect(env["StepEnded"], EnvConfig.OnTrainingStepEnded)
        EventFunctions.Connect(env["EpisodeStarted"], EnvConfig.OnEpisodeStarted)
        EventFunctions.Connect(env["EpisodeEnded"], EnvConfig.OnEpisodeEnded)
        EventFunctions.Connect(env["ProximityDetected"], EnvConfig.OnProximityDetected)
        EnvFunctions.RunTrain(env)

    # Plot the training results.
    EpisodeFunctions.PlotRewards(episodes)
    EpisodeFunctions.PlotSteps(episodes)

    # Draw decision arrows on render.
    EventFunctions.Connect(env["Rendered"], EnvConfig.OnRendered)

    # Disconnect the training events.
    EventFunctions.DisconnectAll(env["StepStarted"])
    EventFunctions.DisconnectAll(env["StepEnded"])
    EventFunctions.DisconnectAll(env["EpisodeStarted"])
    EventFunctions.DisconnectAll(env["EpisodeEnded"])

    # Connect the testing events and view result of training.
    EventFunctions.Connect(env["StepStarted"], EnvConfig.OnTestingStepStarted)
    EnvFunctions.RunTest(env)

    # Save the training results.
    DataStoreFunctions.Save(params, lookups, episodes)