from pennylane import numpy as pnp
from enum import Enum
import random

TAU_GLOBAL = 5e-2   # Dephase tau
PARAS_GLOBAL = pnp.zeros(3)
PHI_GLOBAL  =  0


class DataIndex(Enum):
    BEFORE = 0
    PHI = 0
    CFI = 1
    PARAS = 2
    THETA_X = 2
    PHI_Z = 3