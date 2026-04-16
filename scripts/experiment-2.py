from typing import Any, List
from scripts.datastore import DataStoreFunctions
from scripts.env import EnvFunctions
from scripts.episode import EpisodeFunctions, Episode
from scripts.event import EventFunctions
from scripts.test import EnvTestFunctions, EnvTest


class Experiment2Functions:
    @staticmethod
    def OnTrainingStepStarted(env: EnvTest, message: Any):
        # Clear the last step's actions and rewards.
        env["Actions"].clear()
        env["Rewards"].clear()

        # Update each agent using Q-learning.
        for index, agent in enumerate(env["Env"]["Agents"]):
            # Get the action for the given observation.
            env["Actions"].append(EnvTestFunctions.QAction(env, index, message["State"]))
            # Update the agent and track the given reward.
            env["Rewards"].append(EnvTestFunctions.UpdateAgentDefault(env, agent, env["Actions"][index]))
            # Update the agent's last action for render orientation.
            agent["LastAction"] = env["Actions"][index]

    @staticmethod
    def OnTestingStepStarted(env: EnvTest, message: Any):
        for index, agent in enumerate(env["Env"]["Agents"]):
            agent["LastAction"] = EnvTestFunctions.QAction(env, index, message["State"])
            EnvTestFunctions.UpdateAgentDefault(env, agent, agent["LastAction"])


if __name__ == "__main__":
    file_names = [f"Experiment_2_{index}" for index in range(5)]

    params = {
        "AgentCount": 2,
        "FoodCount": 15,
        "ObstacleCount": 10,
        "NestCount": 1,
        "GridSize": {"X": 15, "Y": 15},
        "Seed": 0,
        "MaxSteps": 5_000,
        "EpisodeCount": 10_000,
        "ProximityRadius": 0
    }

    runs: List[List[Episode]] = []

    for index, file_name in enumerate(file_names):
        params["Seed"] = index
        lookups, episodes = DataStoreFunctions.Load(params, file_name)

        env = EnvTestFunctions.EnvTest(params)
        env["Lookups"] = lookups
        env["Episodes"] = episodes

        EnvFunctions.Init(env["Env"])

        if len(episodes) == 0:
            # Connect the training events and start training.
            EventFunctions.Connect(env["Env"]["StepStarted"], lambda *args, **kwargs: Experiment2Functions.OnTrainingStepStarted(env, *args, **kwargs))
            EventFunctions.Connect(env["Env"]["StepEnded"], lambda *args, **kwargs: EnvTestFunctions.OnTrainingStepEnded(env, *args, **kwargs))
            EventFunctions.Connect(env["Env"]["EpisodeEnded"], lambda *args, **kwargs: EnvTestFunctions.OnEpisodeEnded(env, *args, **kwargs))
            EventFunctions.Connect(env["Env"]["ProximityDetected"], lambda *args, **kwargs: EnvTestFunctions.OnProximityDetectedNoExchange(env, *args, **kwargs))
            EventFunctions.Connect(env["Env"]["MaxStepReached"], lambda *args, **kwargs: EnvTestFunctions.OnMaxStepReached(env))

            # Train the environment
            EnvFunctions.RunTrain(env["Env"])

            # Disconnect the training events.
            EventFunctions.DisconnectAll(env["Env"]["StepStarted"])
            EventFunctions.DisconnectAll(env["Env"]["StepEnded"])
            EventFunctions.DisconnectAll(env["Env"]["EpisodeStarted"])
            EventFunctions.DisconnectAll(env["Env"]["EpisodeEnded"])
            EventFunctions.DisconnectAll(env["Env"]["MaxStepReached"])

            # Save the training results.
            DataStoreFunctions.Save(lookups, episodes, file_name)

        # Connect the testing events and view result of training.
        EventFunctions.Connect(env["Env"]["Rendered"], lambda *args, **kwargs: EnvTestFunctions.OnRendered(env, *args, **kwargs))
        EventFunctions.Connect(env["Env"]["Reset"], lambda *args, **kwargs: EnvTestFunctions.OnReset(env, *args, **kwargs))
        EventFunctions.Connect(env["Env"]["Ticked"], lambda *args, **kwargs: EnvTestFunctions.OnTicked(env, *args, **kwargs))
        EventFunctions.Connect(env["Env"]["StepStarted"], lambda *args, **kwargs: Experiment2Functions.OnTestingStepStarted(env, *args, **kwargs))

        runs.append(episodes)

        env["Epsilon"] = 0 # Set to zero to get the most optimal action.
        # EnvFunctions.RunTest(env["Env"])

    for run, episodes in enumerate(runs):
        print(f"Run: {run}: {EpisodeFunctions.GetEpisodesToConvergence(episodes)}")

    EpisodeFunctions.PlotSteps(runs)
    EpisodeFunctions.PlotRewards(runs)
    EpisodeFunctions.PlotPotentialExchanges(runs)
    EpisodeFunctions.PlotExchanges(runs)