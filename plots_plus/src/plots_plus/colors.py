import matplotlib as mpl
import numpy as np


class Color:
    BLACK = (0, 0, 0)
    BLUE = (55 / 256, 88 / 256, 136 / 256)
    LIGHT_BLUE = (137 / 256, 173 / 256, 220 / 256)
    RED = (161 / 256, 34 / 256, 0)
    GREEN = (141 / 256, 201 / 256, 20 / 256)  # (0, 124/256, 6/256)
    YELLOW = (227 / 256, 193 / 256, 0)
    LIGHT_GREY = (240 / 256, 240 / 256, 240 / 256)
    ORANGE = (247 / 256, 110 / 256, 66 / 256)
    GREY = (200 / 256, 200 / 256, 200 / 256)


LLM_COLOR_MAP = [
    Color.BLUE,
    Color.YELLOW,
    Color.RED,
    Color.GREEN,
]  # TODO check and extend
LLM_ORDER = ["Qwen/Qwen3-32B-AWQ"]  # TODO extend and apply

num_iterations = 5
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "multiline",
    [Color.YELLOW, Color.GREY, Color.BLUE],
    N=num_iterations,  # TODO make adjustable?
)
ITERATION_COLOR_MAP = [cmap(i) for i in np.linspace(0, 1, num_iterations)]

CORRELATION_COLOR_MAP = [Color.BLUE, Color.YELLOW]
REWARD_COLOR_MAP = [
    Color.BLUE,
    Color.YELLOW,
    Color.RED,
    Color.GREEN,
    Color.LIGHT_BLUE,
    Color.BLACK,
    Color.ORANGE,
]

TOKEN_COLOR_MAP = [Color.BLACK, Color.BLUE, Color.RED, Color.ORANGE, Color.YELLOW]
TOKEN_ORDER = [
    "Total Tokens",
    "Prompt Tokens",
    "Completion Tokens",
    "Answer Tokens",
    "Thinking Tokens",
]


# control package visibility
__all__ = [
    "Color",
    "LLM_COLOR_MAP",
    "LLM_ORDER",
    "ITERATION_COLOR_MAP",
    "CORRELATION_COLOR_MAP",
    "REWARD_COLOR_MAP",
    "TOKEN_COLOR_MAP",
    "TOKEN_ORDER",
]


def __dir__():
    return __all__
