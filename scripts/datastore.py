from typing import List, Tuple
from scripts.policy import PolicyLookup, PolicyFunctions
from scripts.episode import Episode
from scripts.env import EnvParams
import dill
import os


class DataStoreFunctions:
    @staticmethod
    def ParamsToFileName(params: EnvParams):
        name = ""
        for key, value in params.items():
            name = name + str(key) + str(value)
        name = name.translate(str.maketrans("", "", "{'} :,"))
        return name

    @staticmethod
    def Load(params: EnvParams) -> Tuple[List[PolicyLookup], List[Episode]]:
        os.makedirs(name="../runs", exist_ok=True)

        try:
            with open(f"../runs/{DataStoreFunctions.ParamsToFileName(params)}.dill", "rb") as file:
                data = dill.load(file)
                lookups: List[PolicyLookup] = data["Lookups"]
                episodes: List[Episode] = data["Episodes"]

        except FileNotFoundError:
            lookups, episodes = [], []
            for _ in range(params["AgentCount"]):
                lookup = PolicyFunctions.PolicyLookup(params["GridSize"], params["FoodCount"])
                lookups.append(lookup)

        return lookups, episodes

    @staticmethod
    def Save(params: EnvParams, lookups: List[PolicyLookup], episodes: List[Episode]) -> None:
        os.makedirs(name="../runs", exist_ok=True)

        with open(f"../runs/{DataStoreFunctions.ParamsToFileName(params)}.dill", "wb") as file:
            dill.dump({
                "Lookups": lookups,
                "Episodes": episodes,
            }, file)

        return None
