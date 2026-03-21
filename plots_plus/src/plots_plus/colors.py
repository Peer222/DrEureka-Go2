import matplotlib as mpl
import numpy as np


class Color:
    BLACK = (0, 0, 0)
    GREY = (180 / 256, 180 / 256, 180 / 256)
    LIGHT_GREY = (220 / 256, 220 / 256, 220 / 256)
    SUBTLE_GREY = (240 / 256, 240 / 256, 240 / 256)

    DARK_BLUE = (55 / 256, 88 / 256, 136 / 256)  ###
    BLUE = (96 / 256, 135 / 256, 191 / 256)
    LIGHT_BLUE = (162 / 256, 192 / 256, 235 / 256)

    DARK_RED = (161 / 256, 34 / 256, 0)
    RED = (236 / 256, 76 / 256, 36 / 256)
    LIGHT_RED = (255 / 256, 135 / 256, 105 / 256)

    DARK_GREEN = (15 / 256, 100 / 256, 15 / 256)
    GREEN = (60 / 256, 190 / 256, 60 / 256)
    LIGHT_GREEN = (166 / 256, 235 / 256, 166 / 256)

    YELLOW = (227 / 256, 193 / 256, 0)  ###
    ORANGE = (247 / 256, 110 / 256, 66 / 256)
    PURPLE = (210 / 256, 170 / 256, 235 / 256)
    PINK = (230 / 256, 77 / 256, 135 / 256)


LLM_COLOR_MAP = [
    Color.DARK_BLUE,
    Color.YELLOW,
    Color.DARK_RED,
    Color.DARK_GREEN,
]  # TODO check and extend
LLM_ORDER = ["Qwen/Qwen3-32B-AWQ"]  # TODO extend and apply

CONTINUOUS_COLOR_MAP = mpl.colors.LinearSegmentedColormap.from_list(
    "continuous-yellow-grey-blue",
    [Color.YELLOW, Color.GREY, Color.DARK_BLUE],
)
num_iterations = 5
ITERATION_COLOR_MAP = [
    CONTINUOUS_COLOR_MAP(i) for i in np.linspace(0, 1, num_iterations)
]

CORRELATION_COLOR_MAP = [Color.DARK_BLUE, Color.YELLOW]
REWARD_COLOR_MAP = [
    Color.DARK_BLUE,
    Color.GREY,
    Color.DARK_GREEN,
    Color.YELLOW,
    Color.DARK_RED,
    Color.LIGHT_GREY,
    Color.GREEN,
    Color.PURPLE,
    Color.LIGHT_BLUE,
    Color.BLACK,
    Color.ORANGE,
    Color.PINK,
]

TOKEN_COLOR_MAP = [
    Color.BLACK,
    Color.DARK_BLUE,
    Color.DARK_RED,
    Color.ORANGE,
    Color.YELLOW,
]
TOKEN_ORDER = [
    "Total Tokens",
    "Prompt Tokens",
    "Completion Tokens",
    "Answer Tokens",
    "Thinking Tokens",
]

VELOCITY_COLOR_MAP = [Color.DARK_BLUE, Color.DARK_RED, Color.GREEN]
JOINT_COLOR_MAP = [
    Color.DARK_RED,
    Color.RED,
    Color.LIGHT_RED,
    Color.DARK_GREEN,
    Color.GREEN,
    Color.LIGHT_GREEN,
    Color.DARK_BLUE,
    Color.BLUE,
    Color.LIGHT_BLUE,
    Color.BLACK,
    Color.GREY,
    Color.LIGHT_GREY,
]
CONTACT_FORCES_COLOR_MAP = [Color.RED, Color.GREEN, Color.BLUE, Color.GREY]


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
    "VELOCITY_COLOR_MAP",
    "JOINT_COLOR_MAP",
    "CONTACT_FORCES_COLOR_MAP",
    "CONTINUOUS_COLOR_MAP",
]


def __dir__():
    return __all__
