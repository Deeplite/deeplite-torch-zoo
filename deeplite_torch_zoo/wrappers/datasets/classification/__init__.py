try:
    from .vww import *
except ImportError:
    pass

from .cifar100 import *
from .imagenet import *
from .mnist import *
