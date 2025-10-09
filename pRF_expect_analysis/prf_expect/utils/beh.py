import numpy as np
import math


def calculate_visual_angle(screen_height, distance_to_screen):
    """
    Calculate the visual angle given the height of the screen and the distance to the screen.

    Parameters:
    screen_height (float): Height of the screen in centimeters.
    distance_to_screen (float): Distance to the screen in centimeters.

    Returns:
    float: Visual angle in degrees.
    """
    # Convert the visual angle using the formula: theta = 2 * atan((h / 2) / d)
    visual_angle = 2 * math.degrees(math.atan((screen_height / 2) / distance_to_screen))
    return visual_angle


# calculate polar angle with x and y
def calculate_polar_angle(x, y):
    """
    Calculate the polar angle given the x and y coordinates.

    Parameters:
    x (float): X coordinate.
    y (float): Y coordinate.

    Returns:
    float: Polar angle in degrees.
    """
    # Convert the polar angle using the formula: theta = arctan(y / x)
    polar_angle = np.arctan2(y, x)
    return polar_angle
