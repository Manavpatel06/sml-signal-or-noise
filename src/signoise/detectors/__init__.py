from signoise.detectors.base import Detector, PointwiseDetector, point_features
from signoise.detectors.gmm import GMMDetector
from signoise.detectors.hmm import HMMDetector
from signoise.detectors.kmeans import KMeansDetector
from signoise.detectors.knn import KNNDetector
from signoise.detectors.ksigma import KSigmaDetector
from signoise.detectors.pca import PCADetector

# name -> class; the harness and CLI build detectors from this table.
REGISTRY = {
    cls.name: cls
    for cls in (KSigmaDetector, KNNDetector, KMeansDetector, GMMDetector,
                PCADetector, HMMDetector)
}

__all__ = ["Detector", "PointwiseDetector", "point_features", "REGISTRY",
           *[c.__name__ for c in REGISTRY.values()]]
