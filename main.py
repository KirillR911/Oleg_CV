import logging

import torch

from utils import *

logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('./models/model.pt', map_location=device)
detect_oleg(Image.open("0_3002.jpg"), model, device, logger)
print(classify_face(extract_faces(Image.open("0_3002.jpg"), logger, "lol", device), model, device, logger))
