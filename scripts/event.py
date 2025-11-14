from typing import List, Callable, TypedDict, Any, Optional


class Event(TypedDict):
    Callbacks: List[Callable]


class EventFunctions:
    @staticmethod
    def Event():
        return {
            "Callbacks": [],
        }

    @staticmethod
    def Connect(event: Event, callback: Callable):
        event["Callbacks"].append(callback)

    @staticmethod
    def Disconnect(event: Event, callback: Callable):
        event["Callbacks"].remove(callback)

    @staticmethod
    def DisconnectAll(event: Event):
        event["Callbacks"] = []

    @staticmethod
    def Fire(event: Event, arg: Any):
        for callback in event["Callbacks"]:
            callback(arg)
