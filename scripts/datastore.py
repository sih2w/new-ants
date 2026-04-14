from typing import Tuple
from scripts.policy import PolicyLookup, PolicyFunctions
from scripts.episode import Episode
from scripts.env import EnvParams
import dill
import os
from typing import List


class DataStoreFunctions:
    @staticmethod
    def Load(params: EnvParams, file_name: str) -> Tuple[List[PolicyLookup], List[Episode]]:
        os.makedirs(name="../runs", exist_ok=True)

        try:
            with open(f"../runs/{file_name}.dill", "rb") as file:
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
    def Save(lookups: List[PolicyLookup], episodes: List[Episode], file_name: str) -> None:

        os.makedirs(name="../runs", exist_ok=True)
        with open(f"../runs/{file_name}.dill", "wb") as file:
            dill.dump({
                "Lookups": lookups,
                "Episodes": episodes,
            }, file)

        return None
