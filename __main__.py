from .cli import main
import warnings
warnings.filterwarnings("ignore")  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# call the CLI handler when the module is executed as `python -m crepe`
main()
