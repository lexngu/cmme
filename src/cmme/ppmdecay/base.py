from enum import Enum


class PPMEscapeMethod(Enum):
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    AX = "ax"


class PPMModelType(Enum):
    SIMPLE = "SIMPLE"
    DECAY = "DECAY"
