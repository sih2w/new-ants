from typing import TypedDict, List
from matplotlib import pyplot as plt


class Episode(TypedDict):
    AverageRewards: List[float]


class EpisodeFunctions:
    TextFontSize = 12
    NumberFontSize = 10
    FontName = "Times New Roman"
    Step = 100
    Style = "dark_background"

    @staticmethod
    def Episode() -> Episode:
        return {
            "AverageRewards": []
        }

    @staticmethod
    def AverageByInterval(numbers: List[float], step: int) -> List[float]:
        averaged_numbers = []

        for start in range(0, len(numbers), step):
            end = start + step
            segment = numbers[start:end]
            average = sum(segment) / float(len(segment)) if segment else 0.0
            averaged_numbers.append(average)

        return averaged_numbers

    @staticmethod
    def PlotRewards(episodes: List[Episode]):
        y = [sum(episode["AverageRewards"]) for episode in episodes]
        y = EpisodeFunctions.AverageByInterval(y, EpisodeFunctions.Step)
        x = [index * EpisodeFunctions.Step for index in range(len(y))]

        plt.style.use(EpisodeFunctions.Style)
        plt.ylabel("Average Rewards", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
        plt.yticks(fontsize=EpisodeFunctions.NumberFontSize, fontname=EpisodeFunctions.FontName)
        plt.title("Rewards per Episode", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
        plt.xlabel(f"Episode", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
        plt.xticks(fontsize=EpisodeFunctions.NumberFontSize, fontname=EpisodeFunctions.FontName)
        plt.plot(x, y)
        plt.show()

    @staticmethod
    def PlotSteps(episodes: List[Episode]):
        y = [len(episode["AverageRewards"]) for episode in episodes]
        y = EpisodeFunctions.AverageByInterval(y, EpisodeFunctions.Step)
        x = [index * EpisodeFunctions.Step for index in range(len(y))]

        plt.style.use(EpisodeFunctions.Style)
        plt.title("Steps per Episode", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
        plt.ylabel("Steps", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
        plt.yticks(fontsize=EpisodeFunctions.NumberFontSize, fontname=EpisodeFunctions.FontName)
        plt.xlabel(f"Episode", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
        plt.xticks(fontsize=EpisodeFunctions.NumberFontSize, fontname=EpisodeFunctions.FontName)
        plt.plot(x, y)
        plt.show()