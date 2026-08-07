from collections import defaultdict


class EventBus:
    """
    简易事件总线

    后续可替换Redis Stream
    """

    def __init__(self):

        self.listeners = defaultdict(list)

    def subscribe(
        self,
        event_name: str,
        callback
    ):

        self.listeners[event_name].append(callback)

    def publish(
        self,
        event_name: str,
        data
    ):

        for callback in self.listeners[event_name]:

            callback(data)