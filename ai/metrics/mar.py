import math


def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two 2D points.
    """

    x1, y1 = point1
    x2, y2 = point2

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def calculate_mar(mouth_points):
    """
    Calculate Mouth Aspect Ratio (MAR).

    Expected point order:

    0 -> left mouth corner
    1 -> upper inner lip
    2 -> right mouth corner
    3 -> lower inner lip

    MAR = vertical mouth opening / mouth width
    """

    left_corner = mouth_points[0]
    upper_lip = mouth_points[1]
    right_corner = mouth_points[2]
    lower_lip = mouth_points[3]

    vertical_distance = euclidean_distance(
        upper_lip,
        lower_lip
    )

    horizontal_distance = euclidean_distance(
        left_corner,
        right_corner
    )

    if horizontal_distance == 0:
        return 0.0

    mar = vertical_distance / horizontal_distance

    return mar