import logging
import os
import sys
from zipfile import ZipFile
import shutil

import pandas as pd
import torch
from tqdm import tqdm

from utils import *


def main():
    logger = logging.getLogger(__name__)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load('./models/model.pt', map_location=device)

    target_unzip_path = "./unziped/"
    if not os.path.exists(target_unzip_path):
        os.makedirs(target_unzip_path)

    file_name = sys.argv[1]

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

    for i in tqdm(all_images):
        name = i
        i = target_unzip_path + i
        label = classify_face(extract_faces(Image.open(i), logger, i,
                                            device), model, device, logger)
        res_d["names"].append(name)
        res_d["labels"].append(label)
    df = pd.DataFrame.from_dict(res_d)
    df.to_csv("./out.csv", index=False)
    if os.path.exists(target_unzip_path):
        shutil.rmtree(target_unzip_path, ignore_errors=True)



if __name__ == '__main__':
    print(sys.argv[1])
    main()
