from abc import ABC, abstractmethod

from ..event import Event


class Backend(ABC):
    @abstractmethod
    def emit(self, event: Event) -> bool: ...
