# Supervised learning + DAL
from .random_sampling import RandomSampling
from .least_confidence import LeastConfidence
from .margin_sampling import MarginSampling
from .entropy_sampling import EntropySampling
from .least_confidence_dropout import LeastConfidenceDropout
from .margin_sampling_dropout import MarginSamplingDropout
from .entropy_sampling_dropout import EntropySamplingDropout
from .kmeans_sampling import KMeansSampling
from .kcenter_greedy import KCenterGreedy
from .bayesian_active_learning_disagreement_dropout import BALDDropout
from .core_set import CoreSet
from .adversarial_bim import AdversarialBIM
from .adversarial_deepfool import AdversarialDeepFool
from .active_learning_by_learning import ActiveLearningByLearning
from .badge_sampling  import BadgeSampling
from .baseline_sampling  import BaselineSampling
from .wasserstein_adversarial import WAAL
from .learning_loss_for_al import LearningLoss
from .vaal import VAAL
from .batch_active_learning_at_scale import ClusterMarginSampling
from .batch_BALD import BatchBALD
from .uncertainGCN import uncertainGCN
from .coreGCN import coreGCN
from .mcadl import MCADL

# Semi-supervise(mixmatch) + DAL
from .ssl_rand import ssl_Random
from .ssl_diff2augkmeans import ssl_Diff2AugKmeans
from .ssl_diff2augdirect import ssl_Diff2AugDirect
from .ssl_consistency import ssl_Consistency
from .ssl_least_confidence import ssl_LC

# Other augmentation/semi-supervise + random selection
from .aug_uda import uda
from .semi_fixmatch import fixmatch
from .semi_flexmatch import flexmatch
from .semi_pseudolabel import pseudolabel
