from model_training.correction_methods.base_correction_method import Vanilla
from model_training.correction_methods.rrr import (
    RRR_CE,
    RRR_CE_L1,
    RRR_ExpMax,
    RRR_ExpMax_L1,
    RRR_ExpTarget
)
from model_training.correction_methods.cdep import CDEP
from model_training.correction_methods.clarc import (
    AClarc,
    AClarcFullFeature,
    PClarc,
    PClarcFullFeature,
    GClarc,
    GClarc2,
    GClarc3,
    GLocClarc,
    GLocClarc2,
    GLocClarc3,
    RClarc,
    RLocClarc
)

def get_correction_method(method_name):
    CORRECTION_METHODS = {
        'Vanilla': Vanilla,
        'CDEP': CDEP,
        'AClarc': AClarc,
        'AClarcFullFeature': AClarcFullFeature,
        'PClarc': PClarc,
        'PClarcFullFeature': PClarcFullFeature,
        'RRR_CE': RRR_CE,
        'RRR_CE_L1': RRR_CE_L1,
        'RRR_ExpMax': RRR_ExpMax,
        'RRR_ExpMax_L1': RRR_ExpMax_L1,
        'RRR_ExpTarget': RRR_ExpTarget,
        'GClarc': GClarc,
        'GClarc2': GClarc2,
        'GClarc3': GClarc3,
        'GLocClarc': GLocClarc,
        'GLocClarc2': GLocClarc2,
        'GLocClarc3': GLocClarc3,
        'RClarc': RClarc,
        'RLocClarc': RLocClarc
    }

    assert method_name in CORRECTION_METHODS.keys(), f"Correction method '{method_name}' unknown, choose one of {list(CORRECTION_METHODS.keys())}"
    return CORRECTION_METHODS[method_name]