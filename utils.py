import logging

import numpy as np
import torch
from facenet_pytorch import MTCNN, fixed_image_standardization
from PIL import Image, ImageDraw
from torchvision import transforms


def extract_faces(img, logger, img_name, device) -> list:
    '''
        Extracts all faces from given Image(PIL.Image.Image)
        Returns python list of PIL.Image.Image objects 
        Uses MTCNN to get boxes for faces
    '''
    mtcnn = MTCNN(keep_all=True, device=device)
    try:
        boxes, _ = mtcnn.detect(img)
    except:
        logger.warning("Could not detect faces on image {}".format(img_name))
        return None
    imgs = []
    if boxes is None:
        return None
    for i in boxes:
        imgs.append(img.crop(i))
        # img.crop(i).show()
    return imgs


def get_batch(samples: list, device, size: int = 32) -> torch.Tensor:
    '''
        Create batch from given data to pass to forward func.
    '''
    while(len(samples) < size):
        samples.append(torch.zeros((3, 256, 256)))
    samples = torch.cat(samples)
    samples = samples.view((32, 3, 256, 256)).to(device)
    return samples


def classify_face(imgs: list, model, device, logger):
    '''
        Classify image if it has Oleg Tinkov
        Gets all faces from image, model, device, logger
        Returns prob of Oleg Tinkov
    '''
    if imgs is None:
        return 0.0
    trans = transforms.Compose([
        transforms.Resize((256, 256)),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    imgs = [trans(img) for img in imgs]
    number_of_faces = len(imgs)
    imgs = get_batch(imgs, device)
    model.eval()
    sg = torch.nn.Sigmoid()
    rez = (sg(model(imgs)))[:number_of_faces]
    ans = max([i[1] for i in rez])
    del rez
    del sg
    return ans.item()


def color_image(imgs: list, model, device, logger):
    '''
        Locates Oleg TInkof ob image
    '''

    trans = transforms.Compose([
        transforms.Resize((256, 256)),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    imgs = [trans(img) for img in imgs]
    number_of_faces = len(imgs)
    imgs = get_batch(imgs, device)
    sg = torch.nn.Sigmoid()
    rez = (sg(model(imgs)))[:number_of_faces]
    olegs_prob = [i[1] for i in rez]
    if max(olegs_prob) > 0.5:
        ans = [0 if i != max(olegs_prob) else 1 for i in olegs_prob]
    else:
        logger.info("No Oleg Tinkov")
        ans = [0] * number_of_faces
    del rez
    del sg
    del olegs_prob
    return ans


def detect_oleg(img, model, device, logger):
    '''
        Drawing Bounding boxes for all faces, Green if Oleg else Red
    '''
    mtcnn = MTCNN(keep_all=True, device=device)
    try:
        boxes, _ = mtcnn.detect(img)
    except:
        logger.warning("[ERROR]  skiping")
        return None
    imgs = []
    for i in boxes:
        imgs.append(img.crop(i))
    ans = color_image(imgs, model, device, logger)
    img_draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        img_draw.rectangle(boxes[i], width=4,
                           outline='red' if ans[i] == 0 else 'green')
    del ans
    img.show()
