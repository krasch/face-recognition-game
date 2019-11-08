import numpy as np
from enum import Enum
from collections import namedtuple

from face_recognition_live.config import CONFIG


class MATCH_TYPE(Enum):
    MATCH = 1
    LIKELY_NO_MATCH = 2
    DEFINITELY_NO_MATCH = 3


MatchResult = namedtuple("MatchResult", ["match_type", "best_match", "matches", "distances"])


def find_best_match(features, stored_faces):
    if len(stored_faces) == 0:
        return MatchResult(MATCH_TYPE.DEFINITELY_NO_MATCH, None, [], [])

    distances = np.array([np.linalg.norm(features - saved_face.features) for saved_face in stored_faces])
    distances2 = [features-saved_face.features for saved_face in stored_faces]
    distances2 = np.array([np.dot(d, d) for d in distances2])
    print(distances, distances2)
    distances = distances2
    best_match = np.argmin(distances)

    config = CONFIG["recognition"]["models"]["matching"]

    matches = [stored_faces[d] for d in np.argsort(distances) if distances[d] < config["match_cutoff"]]
    matches = matches[:config["num_matches"]]
    #print(distances[np.argsort(distances)])

    if distances[best_match] < config["match_cutoff"]:
        return MatchResult(MATCH_TYPE.MATCH, stored_faces[best_match], matches, distances)
    elif distances[best_match] > config["storage_cutoff"]:
        return MatchResult(MATCH_TYPE.DEFINITELY_NO_MATCH, None, matches, distances)
    else:
        return MatchResult(MATCH_TYPE.LIKELY_NO_MATCH, None, matches, distances)




