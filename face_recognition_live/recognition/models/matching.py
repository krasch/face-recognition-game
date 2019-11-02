import numpy as np
from enum import Enum
from collections import namedtuple


class MATCH_TYPE(Enum):
    MATCH = 1
    LIKELY_NO_MATCH = 2
    DEFINITELY_NO_MATCH = 3


MatchResult = namedtuple("MatchResult", ["match_type", "best_match", "all_matches", "distances"])


def find_best_match(features, stored_faces):
    if len(stored_faces) == 0:
        return MatchResult(MATCH_TYPE.DEFINITELY_NO_MATCH, None, [], [])

    distances = np.array([np.linalg.norm(features - saved_face.features) for saved_face in stored_faces])
    best_match = np.argmin(distances)

    all_matches = [stored_faces[d] for d in np.argsort(distances) if distances[d] < 0.6]

    if distances[best_match] < 0.6:
        return MatchResult(MATCH_TYPE.MATCH, stored_faces[best_match], all_matches, distances)
    elif distances[best_match] > 0.6:
        return MatchResult(MATCH_TYPE.DEFINITELY_NO_MATCH, None, all_matches, distances)
    else:
        return MatchResult(MATCH_TYPE.LIKELY_NO_MATCH, None, all_matches, distances)




