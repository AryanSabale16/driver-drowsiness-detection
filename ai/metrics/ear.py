import math


def euclidean_distance(point1, point2):
    """
    Calculate the straight-line distance between two 2D points.
    """

    x1, y1 = point1
    x2, y2 = point2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR).

    eye_points must contain six points:
    P1, P2, P3, P4, P5, P6
    """

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_distance_1 = euclidean_distance(p2, p6)
    vertical_distance_2 = euclidean_distance(p3, p5)

    horizontal_distance = euclidean_distance(p1, p4)

    # Prevent division by zero
    if horizontal_distance == 0:
        return 0.0

    ear = (
        vertical_distance_1 + vertical_distance_2
    ) / (2.0 * horizontal_distance)

    return ear