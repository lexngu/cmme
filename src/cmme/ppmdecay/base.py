from enum import Enum


class EscapeMethod(Enum):
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    AX = "ax"


class ModelType(Enum):
    SIMPLE = "SIMPLE"
    DECAY = "DECAY"
