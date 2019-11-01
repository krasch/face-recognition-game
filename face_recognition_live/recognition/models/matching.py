import numpy as np
from enum import Enum
from collections import namedtuple


class MATCH_TYPE(Enum):
    MATCH = 1
    LIKELY_NO_MATCH = 2
    DEFINITELY_NO_MATCH = 3


MatchResult = namedtuple("MatchResult", ["match_type", "best_match", "distances"])


def find_best_match(features, stored_faces):
    if len(stored_faces) == 0:
        return MatchResult(MATCH_TYPE.DEFINITELY_NO_MATCH, None, [])

    distances = np.array([np.linalg.norm(features - saved_face.features) for saved_face in stored_faces])
    best_match = np.argmin(distances)

    if distances[best_match] < 0.6:
        return MatchResult(MATCH_TYPE.MATCH, stored_faces[best_match], distances)
    elif distances[best_match] > 0.8:
        return MatchResult(MATCH_TYPE.DEFINITELY_NO_MATCH, None, distances)
    else:
        return MatchResult(MATCH_TYPE.LIKELY_NO_MATCH, None, distances)



