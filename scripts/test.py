from typing import List, Any
from scripts.env import EnvFunctions, Env, EnvParams, Agent, EnvState, AGENT_ACTIONS
from scripts.datastore import DataStoreFunctions
from scripts.event import EventFunctions
from scripts.episode import Episode, EpisodeFunctions
from scripts.policy import PolicyFunctions, PolicyLookup
from scripts.vector import Vector2


class EnvConfig:
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
        EnvConfig.Actions.clear()
        EnvConfig.Rewards.clear()

        for index, agent in enumerate(EnvConfig.Env["Agents"]):
            EnvConfig.Actions.append(EnvConfig.QAction(index, message["State"]))
            EnvConfig.Rewards.append(EnvConfig.UpdateAgent(agent, EnvConfig.Actions[index]))
            agent["LastAction"] = EnvConfig.Actions[index]

    @staticmethod
    def OnTrainingStepEnded(message: Any):
        total_rewards, count = 0, 1

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

        EnvConfig.CurrentEpisode["AverageRewards"].append(total_rewards / count)
        EnvConfig.Epsilon -= EnvConfig.DecayRate

    @staticmethod
    def OnTestingStepStarted(message: Any):
        EnvConfig.Epsilon = 0

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
        EnvConfig.Episodes.append(EnvConfig.CurrentEpisode)
        EnvConfig.CurrentEpisode = EpisodeFunctions.Episode()


if __name__ == "__main__":
    params: EnvParams = {
        "AgentCount": 1,
        "FoodCount": 5,
        "ObstacleCount": 10,
        "NestCount": 1,
        "GridSize": {"X": 10, "Y": 10},
        "Seed": 1,
        "MaxSteps": 10_000,
        "EpisodeCount": 100
    }

    lookups, episodes = DataStoreFunctions.Load(params)
    env: Env = EnvFunctions.Env(params)

    EnvConfig(lookups, episodes, params["EpisodeCount"], env)
    EnvFunctions.PygameInit()
    EnvFunctions.EnvInit(env)
    EventFunctions.Connect(env["Rendered"], EnvConfig.OnRendered)

    if len(episodes) == 0:
        EventFunctions.Connect(env["StepStarted"], EnvConfig.OnTrainingStepStarted)
        EventFunctions.Connect(env["StepEnded"], EnvConfig.OnTrainingStepEnded)
        EventFunctions.Connect(env["EpisodeStarted"], EnvConfig.OnEpisodeStarted)
        EventFunctions.Connect(env["EpisodeEnded"], EnvConfig.OnEpisodeEnded)
        env["RenderStep"] = False
        EnvFunctions.RunAutomatic(env)

    EpisodeFunctions.PlotRewards(episodes)

    env["RenderStep"] = True
    EventFunctions.DisconnectAll(env["StepStarted"])
    EventFunctions.DisconnectAll(env["StepEnded"])
    EventFunctions.DisconnectAll(env["EpisodeStarted"])
    EventFunctions.DisconnectAll(env["EpisodeEnded"])
    EventFunctions.Connect(env["StepStarted"], EnvConfig.OnTestingStepStarted)
    EnvFunctions.RunManual(env)

    DataStoreFunctions.Save(params, lookups, episodes)