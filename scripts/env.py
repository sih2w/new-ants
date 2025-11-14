from typing import List, TypedDict, Literal, Callable, TypeVar, Optional, Tuple
from pygame import Color, Surface
from pygame.font import Font
from pygame.time import Clock
from numpy.random import Generator, PCG64
from vector import Vector2
from scripts.event import Event, EventFunctions
from tqdm import tqdm
import pygame


AGENT_ACTIONS = (
    {
        "Direction": {"X": 0, "Y": -1},
        "Rotation": 0
    },
    {
        "Direction": {"X": 0, "Y": 1},
        "Rotation": 180
    },
    {
        "Direction": {"X": -1, "Y": 0},
        "Rotation": 90
    },
    {
        "Direction": {"X": 1, "Y": 0},
        "Rotation": -90
    }
)

IMAGE_PIXEL_WIDTH = 40
ARROW_IMAGE = "../images/icons8-triangle-48.png"
AGENT_IMAGE = "../images/icons8-ant-48.png"
NEST_IMAGE = "../images/icons8-egg-basket-48.png"
FOOD_IMAGE = "../images/icons8-whole-apple-48.png"
CARRIED_FOOD_IMAGE = "../images/icons8-whole-apple-carried-48.png"

T = TypeVar("T")
E = TypeVar("E")


class Food(TypedDict):
    Location: Vector2
    Status: Literal["Carried", "Deposited", "Dropped"]
    SpawnLocation: Vector2


class Obstacle(TypedDict):
    Location: Vector2


class Nest(TypedDict):
    Location: Vector2


class Agent(TypedDict):
    Location: Vector2
    Food: List[Food]
    LastAction: int
    SpawnLocation: Vector2
    Capacity: int
    Color: Color


class EnvState(TypedDict):
    AgentLocations: List[Vector2]
    CarryingFood: List[bool]
    FoodDeposited: int


class Env(TypedDict):
    Initialized: bool
    Agents: List[Agent]
    Reset: Event
    AllFoodDeposited: Event
    MaxStepReached: Event
    StepStarted: Event
    StepEnded: Event
    Rendered: Event
    EpisodeStarted: Event
    EpisodeEnded: Event
    Food: List[Food]
    Obstacles: List[Obstacle]
    Nests: List[Nest]
    Generator: Generator
    GridSize: Vector2
    Window: Optional[Surface]
    WindowSize: Vector2
    Clock: Optional[Clock]
    Font: Optional[Font]
    Running: bool
    RenderStep: bool
    CurrentStep: int
    MaxSteps: int
    EpisodeCount: int


class EnvParams(TypedDict):
    AgentCount: int
    FoodCount: int
    ObstacleCount: int
    NestCount: int
    GridSize: Vector2
    Seed: int
    MaxSteps: int
    EpisodeCount: int


class EnvFunctions:
    @staticmethod
    def AgentColor(key: int):
        color = Color(0)
        color.hsla = (360.00 / (key + 1.00), 100.00, 50.00, 100.00)
        return color

    @staticmethod
    def UpdateCarriedFoodLocations(env: Env):
        for agent in env["Agents"]:
            for food in agent["Food"]:
                food["Location"] = agent["Location"]
        return None

    @staticmethod
    def Agent(key: int) -> Agent:
        return {
            "Location": {"X": 0, "Y": 0},
            "Food": [],
            "LastAction": 0,
            "SpawnLocation": {"X": 0, "Y": 0},
            "Capacity": 1,
            "Color": EnvFunctions.AgentColor(key)
        }

    @staticmethod
    def Food() -> Food:
        return {
            "Location": {"X": 0, "Y": 0},
            "Status": "Dropped",
            "SpawnLocation": {"X": 0, "Y": 0},
        }

    @staticmethod
    def Obstacle() -> Obstacle:
        return {
            "Location": {"X": 0, "Y": 0},
        }

    @staticmethod
    def Nest() -> Obstacle:
        return {
            "Location": {"X": 0, "Y": 0},
        }

    @staticmethod
    def Env(params: EnvParams) -> Env:
        return {
            "Initialized": False,
            "Agents": [EnvFunctions.Agent(key) for key in range(params["AgentCount"])],
            "Food": [EnvFunctions.Food() for _ in range(params["FoodCount"])],
            "Obstacles": [EnvFunctions.Obstacle() for _ in range(params["ObstacleCount"])],
            "Nests": [EnvFunctions.Nest() for _ in range(params["NestCount"])],
            "Generator": Generator(PCG64(params["Seed"])),
            "Reset": EventFunctions.Event(),
            "StepEnded": EventFunctions.Event(),
            "StepStarted": EventFunctions.Event(),
            "AllFoodDeposited": EventFunctions.Event(),
            "MaxStepReached": EventFunctions.Event(),
            "Rendered": EventFunctions.Event(),
            "EpisodeStarted": EventFunctions.Event(),
            "EpisodeEnded": EventFunctions.Event(),
            "GridSize": params["GridSize"],
            "Running": False,
            "RenderStep": True,
            "CurrentStep": 0,
            "MaxSteps": params["MaxSteps"],
            "EpisodeCount": params["EpisodeCount"],
            "WindowSize": {
                "X": IMAGE_PIXEL_WIDTH * params["GridSize"]["X"],
                "Y": IMAGE_PIXEL_WIDTH * params["GridSize"]["Y"]
            },
        }

    @staticmethod
    def IsLocationEmpty(env: Env, location: Vector2) -> bool:
        for agent in env["Agents"]:
            if agent["Location"]["X"] == location["X"] and agent["Location"]["Y"] == location["Y"]:
                return False

        for obstacle in env["Obstacles"]:
            if obstacle["Location"]["X"] == location["X"] and obstacle["Location"]["Y"] == location["Y"]:
                return False

        for nest in env["Nests"]:
            if nest["Location"]["X"] == location["X"] and nest["Location"]["Y"] == location["Y"]:
                return False
        return True

    @staticmethod
    def GetEmptyLocation(env: Env) -> Vector2 or None:
        location: Vector2 = {
            "X": env["Generator"].integers(low=0, high=env["GridSize"]["X"]),
            "Y": env["Generator"].integers(low=0, high=env["GridSize"]["Y"]),
        }

        if EnvFunctions.IsLocationEmpty(env, location):
            return location

        for x in range(env["GridSize"]["X"]):
            for y in range(env["GridSize"]["Y"]):
                location: Vector2 = {"X": x, "Y": y}
                if EnvFunctions.IsLocationEmpty(env, location):
                    return location
        return None

    @staticmethod
    def PygameInit():
        if not pygame.get_init():
            pygame.init()
            pygame.display.set_caption("Ants")
            pygame.display.set_icon(pygame.image.load(AGENT_IMAGE))
            pygame.event.set_blocked([
                pygame.MOUSEMOTION,
                pygame.WINDOWENTER,
                pygame.WINDOWLEAVE
            ])
            pygame.event.set_allowed([
                pygame.KEYDOWN,
                pygame.KEYUP
            ])

    @staticmethod
    def EnvInit(env: Env):
        if not env["Initialized"]:
            env["Initialized"] = True
            env["Window"] = pygame.display.set_mode((env["WindowSize"]["X"], env["WindowSize"]["Y"]))
            env["Font"] = pygame.font.SysFont("arialblack", 30)
            env["Clock"] = Clock()

            for agent in env["Agents"]:
                agent["SpawnLocation"] = EnvFunctions.GetEmptyLocation(env)
                agent["Location"] = agent["SpawnLocation"]

            for obstacle in env["Obstacles"]:
                obstacle["SpawnLocation"] = EnvFunctions.GetEmptyLocation(env)
                obstacle["Location"] = obstacle["SpawnLocation"]

            for nest in env["Nests"]:
                nest["SpawnLocation"] = EnvFunctions.GetEmptyLocation(env)
                nest["Location"] = nest["SpawnLocation"]

            for food in env["Food"]:
                food["SpawnLocation"] = EnvFunctions.GetEmptyLocation(env)
                food["Location"] = food["SpawnLocation"]

    @staticmethod
    def EnvReset(env: Env):
        env["CurrentStep"] = 0

        for agent in env["Agents"]:
            agent["Location"] = agent["SpawnLocation"]

        for obstacle in env["Obstacles"]:
            obstacle["Location"] = obstacle["SpawnLocation"]

        for nest in env["Nests"]:
            nest["Location"] = nest["SpawnLocation"]

        for food in env["Food"]:
            food["Location"] = food["SpawnLocation"]
            food["Status"] = "Dropped"

        EventFunctions.Fire(env["Reset"], None)
        return None

    @staticmethod
    def OutOfBounds(env: Env, location: Vector2) -> bool:
        return location["X"] < 0 or location["X"] >= env["GridSize"]["X"] or location["Y"] < 0 or location["Y"] >= env["GridSize"]["Y"]

    @staticmethod
    def InsideObstacle(env: Env, location: Vector2) -> bool:
        for obstacle in env["Obstacles"]:
            if obstacle["Location"]["X"] == location["X"] and obstacle["Location"]["Y"] == location["Y"]:
                return True
        return False

    @staticmethod
    def OnDroppedFood(env: Env, location: Vector2) -> Optional[Food]:
        for food in env["Food"]:
            if food["Location"]["X"] == location["X"] and food["Location"]["Y"] == location["Y"]:
                if food["Status"] == "Dropped":
                    return food
        return None

    @staticmethod
    def OnNest(env: Env, location: Vector2) -> Optional[Nest]:
        for nest in env["Nests"]:
            if nest["Location"]["X"] == location["X"] and nest["Location"]["Y"] == location["Y"]:
                return nest
        return None

    @staticmethod
    def AtCapacity(agent: Agent) -> bool:
        return len(agent["Food"]) >= agent["Capacity"]

    @staticmethod
    def CanPickup(agent: Agent, food: Food) -> bool:
        return not EnvFunctions.AtCapacity(agent) and food["Status"] == "Dropped"

    @staticmethod
    def GiveFood(agent: Agent, food: Food) -> bool:
        if EnvFunctions.CanPickup(agent, food):
            food["Status"] = "Carried"
            agent["Food"].append(food)
            return True
        return False

    @staticmethod
    def CanDeposit(env: Env, agent: Agent, food: Food) -> bool:
        if food in agent["Food"] and EnvFunctions.OnNest(env, agent["Location"]):
            return True
        return False

    @staticmethod
    def Deposit(env: Env, agent: Agent, food: Food) -> bool:
        if EnvFunctions.CanDeposit(env, agent, food):
            food["Status"] = "Deposited"
            agent["Food"].remove(food)
            return True
        return False

    @staticmethod
    def DrawGrass(env: Env, surface: Surface):
        for x in range(env["GridSize"]["X"]):
            for y in range(env["GridSize"]["Y"]):
                pygame.draw.rect(
                    surface=surface,
                    color=(5, 144, 51) if (x % 2 == 0 and y % 2 == 1) or (x % 2 == 1 and y % 2 == 0) else (53, 94, 59),
                    rect=(
                        x * IMAGE_PIXEL_WIDTH,
                        y * IMAGE_PIXEL_WIDTH,
                        IMAGE_PIXEL_WIDTH,
                        IMAGE_PIXEL_WIDTH
                    ),
                )

    @staticmethod
    def DrawNests(env: Env, surface: Surface):
        for nest in env["Nests"]:
            surface.blit(
                pygame.image.load(NEST_IMAGE),
                EnvFunctions.GetDrawPosition(nest["Location"])
            )

    @staticmethod
    def DrawObstacles(env: Env, surface: Surface):
        for obstacle in env["Obstacles"]:
            pygame.draw.rect(
                surface=surface,
                color=(100, 149, 237),
                rect=(
                    obstacle["Location"]["X"] * IMAGE_PIXEL_WIDTH,
                    obstacle["Location"]["Y"] * IMAGE_PIXEL_WIDTH,
                    IMAGE_PIXEL_WIDTH,
                    IMAGE_PIXEL_WIDTH
                ),
            )

    @staticmethod
    def ChangeColor(image: Surface, color: Color):
        surface = pygame.Surface(image.get_size())
        surface.fill(color)
        newImage = image.copy()
        newImage.blit(surface, (0, 0), special_flags=pygame.BLEND_MULT)
        return newImage

    @staticmethod
    def DrawAgents(env: Env, surface: Surface):
        for index, agent in enumerate(env["Agents"]):
            image = pygame.image.load(AGENT_IMAGE)
            image = EnvFunctions.ChangeColor(image, agent["Color"])
            image = pygame.transform.rotate(image, AGENT_ACTIONS[agent["LastAction"]]["Rotation"])
            surface.blit(image, EnvFunctions.GetDrawPosition(agent["Location"]))

    @staticmethod
    def DrawFood(env: Env, surface: Surface):
        for index, food in enumerate(env["Food"]):
            if food["Status"] == "Deposited":
                continue

            position = EnvFunctions.GetDrawPosition(food["Location"])
            if food["Status"] == "Dropped":
                image = pygame.image.load(FOOD_IMAGE)
                surface.blit(image, position)
                surface.blit(
                    env["Font"].render(f"{index}", True, (255, 255, 255)),
                    position
                )
            else:
                image = pygame.image.load(CARRIED_FOOD_IMAGE)
                surface.blit(image, position)

    @staticmethod
    def DrawArrows(env: Env, callback: Callable[[int, Vector2], int], surface: Surface):
        for x in range(env["GridSize"]["X"]):
            for y in range(env["GridSize"]["Y"]):
                for index, agent in enumerate(env["Agents"]):
                    location: Vector2 = {"X": x, "Y": y}
                    action = callback(index, location)

                    image = pygame.image.load(ARROW_IMAGE)
                    image = EnvFunctions.ChangeColor(image, agent["Color"])
                    image = pygame.transform.rotate(image, AGENT_ACTIONS[action]["Rotation"])
                    surface.blit(image, EnvFunctions.GetDrawPosition(location))

    @staticmethod
    def EnvDraw(env: Env, surface: Surface):
        if pygame.get_init():
            EnvFunctions.DrawGrass(env, surface)
            EnvFunctions.DrawObstacles(env, surface)
            EnvFunctions.DrawNests(env, surface)
            EnvFunctions.DrawAgents(env, surface)
            EnvFunctions.DrawFood(env, surface)

    @staticmethod
    def GetDrawPosition(location: Vector2) -> Tuple[float, float]:
        return (
            location["X"] * IMAGE_PIXEL_WIDTH + IMAGE_PIXEL_WIDTH / 2 - IMAGE_PIXEL_WIDTH / 2,
            location["Y"] * IMAGE_PIXEL_WIDTH + IMAGE_PIXEL_WIDTH / 2 - IMAGE_PIXEL_WIDTH / 2,
        )

    @staticmethod
    def AllDeposited(env: Env):
        for food in env["Food"]:
            if food["Status"] != "Deposited":
                return False
        return True

    @staticmethod
    def TryMoveAgent(env: Env, agent: Agent, action: int) -> bool:
        location: Vector2 = {
            "X": agent["Location"]["X"] + AGENT_ACTIONS[action]["Direction"]["X"],
            "Y": agent["Location"]["Y"] + AGENT_ACTIONS[action]["Direction"]["Y"],
        }

        if EnvFunctions.OutOfBounds(env, location):
            return False

        if EnvFunctions.InsideObstacle(env, location):
            return False

        agent["Location"] = location

        return True

    @staticmethod
    def GetState(env: Env) -> EnvState:
        deposited = 0
        for food in env["Food"]:
            if food["Status"] == "Deposited":
                deposited += 1

        return {
            "AgentLocations": [agent["Location"] for agent in env["Agents"]],
            "CarryingFood": [len(agent["Food"]) > 0 for agent in env["Agents"]],
            "FoodDeposited": deposited,
        }

    @staticmethod
    def EnvStep(env: Env):
        old_state = EnvFunctions.GetState(env)
        EventFunctions.Fire(env["StepStarted"], {
            "State": old_state,
        })

        EnvFunctions.UpdateCarriedFoodLocations(env)
        if EnvFunctions.AllDeposited(env):
            EventFunctions.Fire(env["AllFoodDeposited"], None)

        env["CurrentStep"] += 1
        if env["CurrentStep"] >= env["MaxSteps"]:
            EventFunctions.Fire(env["MaxStepReached"], None)

        new_state = EnvFunctions.GetState(env)
        EventFunctions.Fire(env["StepEnded"], {
            "OldState": old_state,
            "NewState": new_state,
        })

        return None

    @staticmethod
    def RenderFrame(env: Env):
        if pygame.get_init():
            surface = pygame.Surface((env["WindowSize"]["X"], env["WindowSize"]["Y"]))
            EnvFunctions.EnvDraw(env, surface)
            EventFunctions.Fire(env["Rendered"], {
                "Surface": surface,
                "State": EnvFunctions.GetState(env),
            })

            env["Window"].blit(surface, surface.get_rect())
            pygame.event.pump()
            pygame.display.flip()

    @staticmethod
    def RunAutomatic(env: Env):
        env["Running"] = True
        EnvFunctions.EnvReset(env)
        EnvFunctions.RenderFrame(env)

        progress_bar = tqdm(total=env["EpisodeCount"])

        for episode in range(env["EpisodeCount"]):
            EnvFunctions.EnvReset(env)
            EventFunctions.Fire(env["EpisodeStarted"], {
                "Episode": episode,
            })

            while not EnvFunctions.AllDeposited(env):
                if not env["Running"] or not pygame.get_init():
                    return

                event = pygame.event.poll()
                EnvFunctions.EnvStep(env)
                if env["RenderStep"]:
                    EnvFunctions.RenderFrame(env)

                if event.type == pygame.QUIT:
                    env["Running"] = False
                    EnvFunctions.Close()

            EventFunctions.Fire(env["EpisodeEnded"], {
                "Episode": episode,
            })

            progress_bar.update(1)
        progress_bar.close()

        return None

    @staticmethod
    def RunManual(env: Env):
        env["Running"] = True
        EnvFunctions.EnvReset(env)
        EnvFunctions.RenderFrame(env)

        while env["Running"] and pygame.get_init():
            event = pygame.event.wait(0)
            if EnvFunctions.AllDeposited(env):
                EnvFunctions.EnvReset(env)

            if event.type == pygame.KEYDOWN:
                if pygame.key.get_pressed()[pygame.K_SPACE]:
                    EnvFunctions.EnvStep(env)
                    if env["RenderStep"]:
                        EnvFunctions.RenderFrame(env)

            if event.type == pygame.QUIT:
                env["Running"] = False
                EnvFunctions.Close()

    @staticmethod
    def HasQuit():
        if pygame.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
        return False

    @staticmethod
    def Close():
        pygame.display.quit()
        pygame.quit()
