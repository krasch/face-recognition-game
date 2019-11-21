import numpy as np
from enum import Enum
from collections import namedtuple

from face_recognition_live.config import CONFIG
from face_recognition_live.monitoring import ProbabilitiesMonitor


class MatchQuality(Enum):
    EXCELLENT = 1
    GOOD = 2
    POOR = 3
    NO_MATCH = 4


MatchingFace = namedtuple("MatchingFace", ["face", "distance", "quality"])


class MATCHING_METHOD(Enum):
    DOT = 1
    L2 = 2


def dot_product(features, saved_features):
    diff = features - saved_features
    return np.dot(diff, diff)


def l2_norm(features, saved_features):
    return np.linalg.norm(features - saved_features)


def get_match_quality(distance, config):
    if distance <= config["excellent_match_cutoff"]:
        return MatchQuality.EXCELLENT
    elif distance <= config["good_match_cutoff"]:
        return MatchQuality.GOOD
    elif distance <= config["poor_match_cutoff"]:
        return MatchQuality.POOR
    else:
        return MatchQuality.NO_MATCH


def init_matching(matching_method):
    if matching_method == MATCHING_METHOD.DOT:
        calculate_distance = dot_product
    elif matching_method == MATCHING_METHOD.L2:
        calculate_distance = l2_norm
    else:
        raise NotImplementedError()

    monitor = ProbabilitiesMonitor()

    def find_best_match(features, known_faces):
        config = CONFIG["recognition"]["models"]["matching"]

        distances = [calculate_distance(features, face.features) for face in known_faces]
        order = np.argsort(distances)

        matches = [MatchingFace(known_faces[d], distances[d], get_match_quality(distances[d], config)) for d in order]
        monitor.add(matches)
        return matches

    return find_best_match


