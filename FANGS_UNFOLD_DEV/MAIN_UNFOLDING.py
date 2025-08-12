import numpy as np
import json
import time
import os
import sys

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

start_time = time.time()

from UTILS import Utils
from MATRIX_BUILDER import *
from METHODS import *
from POSTPROCESS import PostProcessor
from SOLVERFACTORY import SolverFactory
