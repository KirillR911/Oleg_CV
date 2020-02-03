import logging

import torch

from utils import *

logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('./models/model12.pt', map_location=device)
detect_oleg(Image.open("photo_2020-02-03 17.59.26.jpeg"), model, device, logger)
print(classify_face(extract_faces(Image.open("photo_2020-02-03 17.59.26.jpeg"), logger, "lol", device), model, device, logger))
