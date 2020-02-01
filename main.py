import logging

import torch

from utils import *

logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('model.pt', map_location=device)
detect_oleg(Image.open("881.jpg"), model, device, logger)
