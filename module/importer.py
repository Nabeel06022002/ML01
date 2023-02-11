#encoding:utf-8

import glob
import os as os
import os.path as osp
import random
import numpy as np
import json
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')# AGG(Anti-Grain Geometry engine)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models,transforms
import torch.nn.init as init
from torch.autograd import Function
import torch.nn.functional as F

import xml.etree.ElementTree as ET
from itertools import product
from math import sqrt
import time
import librosa
import soundfile as sf
import hashlib






