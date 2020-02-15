import logging
import os
import sys
from zipfile import ZipFile
import shutil
import argparse

import pandas as pd
import torch
from tqdm import tqdm

from utils import *


def main(args):
    logger = logging.getLogger(__name__)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load('./models/model13.pt', map_location=device)

    target_unzip_path = "./unziped/"
    if not os.path.exists(target_unzip_path):
        os.makedirs(target_unzip_path)

    file_name = args.input

    with ZipFile(file_name, 'r') as zip:
        zip.printdir()
        print('Extracting all the files now...')
        zip.extractall(target_unzip_path)
        print('Done!')

    all_images = [
        x for x in os.listdir(target_unzip_path)
        if os.path.isfile(os.path.join(target_unzip_path, x))
    ]
    res_d = {"names": [],
             "labels": []
             }

    for i in (all_images):
        name = i
        i = target_unzip_path + i
        label = classify_face(extract_faces(Image.open(i), logger, i,
                                            device), model, device, logger)
        if args.debug:
            detect_oleg(Image.open(i), model, device, logger)
        res_d["names"].append(name)
        res_d["labels"].append(label)
        print(name,' ',label)
    df = pd.DataFrame.from_dict(res_d)
    df.to_csv("./out.csv", index=False)
    if os.path.exists(target_unzip_path):
        shutil.rmtree(target_unzip_path, ignore_errors=True)

 

if __name__ == '__main__':
    help_msg = "Python zip_process.py --input zip_fn --debug True to show inages with bounding boxes else False"
    parser = argparse.ArgumentParser(description = "Tool to get prob of existing Tinkov on image.", usage = help_msg)
    help = ".Zip filename with .jpg images."
    parser.add_argument("--input", "-i", help = help, required = True)
    parser.add_argument("--debug", "-d", help = "Debugging mode. Displaing images with colored bounding boxes.", default= True, type = bool)
    args = parser.parse_args()
    main(args)
