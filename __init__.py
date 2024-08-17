from .version import version as __version__
from .core import get_activation, predict, process_file
import warnings
warnings.filterwarnings("ignore")  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
