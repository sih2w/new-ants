from typing import TypedDict, List
from matplotlib import pyplot as plt


class Episode(TypedDict):
    TotalRewards: int
    Steps: int
    Exchanges: int
    PotentialExchanges: int
    TerminatedEarly: bool


class EpisodeFunctions:
    TextFontSize = 20
    NumberFontSize = 15
    FontName = "Calibri"
    Padding = 4
    Step = 100

    @staticmethod
    def Episode() -> Episode:
        return {
            "TotalRewards": 0,
            "Steps": 0,
            "Exchanges": 0,
            "PotentialExchanges": 0,
            "TerminatedEarly": False,
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
    def PlotExchanges(runs: List[List[Episode]]):
        for run, episodes in enumerate(runs):
            y = [episode["Exchanges"] for episode in episodes]
            y = EpisodeFunctions.AverageByInterval(y, EpisodeFunctions.Step)
            x = [index * EpisodeFunctions.Step for index in range(len(y))]

            plt.ylabel("Exchanges", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
            plt.yticks(fontsize=EpisodeFunctions.NumberFontSize, fontname=EpisodeFunctions.FontName)
            plt.title("Exchanges per Episode", fontsize=EpisodeFunctions.TextFontSize,
                      fontname=EpisodeFunctions.FontName)
            plt.xlabel(f"Episode", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
            plt.xticks(fontsize=EpisodeFunctions.NumberFontSize, fontname=EpisodeFunctions.FontName)
            plt.tight_layout(pad=EpisodeFunctions.Padding)
            plt.plot(x, y, label=f"Run {run}")

        plt.legend()
        plt.show()

    @staticmethod
    def PlotPotentialExchanges(runs: List[List[Episode]]):
        for run, episodes in enumerate(runs):
            y = [episode["PotentialExchanges"] for episode in episodes]
            y = EpisodeFunctions.AverageByInterval(y, EpisodeFunctions.Step)
            x = [index * EpisodeFunctions.Step for index in range(len(y))]

            plt.ylabel("Potential Exchanges", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
            plt.yticks(fontsize=EpisodeFunctions.NumberFontSize, fontname=EpisodeFunctions.FontName)
            plt.title("Potential Exchanges per Episode", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
            plt.xlabel(f"Episode", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
            plt.xticks(fontsize=EpisodeFunctions.NumberFontSize, fontname=EpisodeFunctions.FontName)
            plt.tight_layout(pad=EpisodeFunctions.Padding)
            plt.plot(x, y, label=f"Run {run}")

        plt.legend()
        plt.show()

    @staticmethod
    def PlotRewards(runs: List[List[Episode]]):
        for run, episodes in enumerate(runs):
            y = [episode["TotalRewards"] for episode in episodes]
            y = EpisodeFunctions.AverageByInterval(y, EpisodeFunctions.Step)
            x = [index * EpisodeFunctions.Step for index in range(len(y))]

            plt.ylabel("Total Rewards", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
            plt.yticks(fontsize=EpisodeFunctions.NumberFontSize, fontname=EpisodeFunctions.FontName)
            plt.title("Rewards per Episode", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
            plt.xlabel(f"Episode", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
            plt.xticks(fontsize=EpisodeFunctions.NumberFontSize, fontname=EpisodeFunctions.FontName)
            plt.tight_layout(pad=EpisodeFunctions.Padding)
            plt.plot(x, y, label=f"Run {run}")

        plt.legend()
        plt.show()

    @staticmethod
    def PlotSteps(runs: List[List[Episode]]):
        for run, episodes in enumerate(runs):
            y = [episode["Steps"] for episode in episodes]
            y = EpisodeFunctions.AverageByInterval(y, EpisodeFunctions.Step)
            x = [index * EpisodeFunctions.Step for index in range(len(y))]

            plt.title("Steps per Episode", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
            plt.ylabel("Steps", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
            plt.yticks(fontsize=EpisodeFunctions.NumberFontSize, fontname=EpisodeFunctions.FontName)
            plt.xlabel(f"Episode", fontsize=EpisodeFunctions.TextFontSize, fontname=EpisodeFunctions.FontName)
            plt.xticks(fontsize=EpisodeFunctions.NumberFontSize, fontname=EpisodeFunctions.FontName)
            plt.tight_layout(pad=EpisodeFunctions.Padding)
            plt.plot(x, y, label=f"Run {run}")

        plt.legend()
        plt.show()