from typing import TypedDict, List
from matplotlib import pyplot as plt


class Episode(TypedDict):
    AverageRewards: List[float]
    Exchanges: List[int]


class EpisodeFunctions:
    @staticmethod
    def Episode() -> Episode:
        return {
            "AverageRewards": [],
            "Exchanges": [],
        }

    @staticmethod
    def PlotRewards(episodes: List[Episode]):
        x = [index for index in range(len(episodes))]
        y = [episode["AverageRewards"][-1] for episode in episodes]
        plt.plot(x, y)
        plt.title("Rewards per Episode")
        plt.ylabel("Reward")
        plt.xlabel(f"Episode")
        plt.show()