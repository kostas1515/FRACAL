import sys
sys.path.insert(1, '../') 
import numpy as np
import pandas as pd 
from tqdm import tqdm
from multiprocessing import Pool
from scipy import optimize
import argparse 
from lvis import LVIS
from pycocotools.coco import COCO


def main(args):
    annfile =args.path
    if args.dset_name !='coco':
        lvis=LVIS(annfile)
        img_ids = lvis.get_img_ids()
        xmin=[]
        ymin=[]
        xmax=[]
        ymax=[]
        width=[]
        height=[]
        category=[]
        image_id=[]
        for id in tqdm(img_ids):
            img=lvis.load_imgs([id])[0]
            w=img['width']
            h=img['height']
            ann_ids = lvis.get_ann_ids([id])
            annotations = lvis.load_anns(ann_ids) 
            for ann in annotations:
                x0 = ann['bbox'][0]/w
                y0= ann['bbox'][1]/h
                x1 = x0 + ann['bbox'][2]/w
                y1 = y0 + ann['bbox'][3]/h
                xmin.append(x0)
                ymin.append(y0)
                xmax.append(x1)
                ymax.append(y1)
                width.append(ann['bbox'][2]/w)
                height.append(ann['bbox'][3]/h)
                category.append(ann['category_id'])
                image_id.append(id)

        df = pd.DataFrame()
        df['xmin'] = xmin
        df['ymin'] = ymin
        df['xmax'] = xmax
        df['ymax'] = ymax
        df['width'] = width
        df['height'] = height
        df['category'] = category
        df['image_id'] = image_id

        df['aspect_ratio'] = df['width']/df['height']
        df['area'] = df['width']*df['height']

        df.to_csv(args.output)    
    else:
        lvis=COCO(annfile)
        img_ids = lvis.getImgIds()
        xmin=[]
        ymin=[]
        xmax=[]
        ymax=[]
        width=[]
        height=[]
        category=[]
        image_id=[]
        for id in tqdm(img_ids):
            img=lvis.loadImgs([id])[0]
            w=img['width']
            h=img['height']
            ann_ids = lvis.getAnnIds([id])
            annotations = lvis.loadAnns(ann_ids) 
            for ann in annotations:
                x0 = ann['bbox'][0]/w
                y0= ann['bbox'][1]/h
                x1 = x0 + ann['bbox'][2]/w
                y1 = y0 + ann['bbox'][3]/h
                xmin.append(x0)
                ymin.append(y0)
                xmax.append(x1)
                ymax.append(y1)
                width.append(ann['bbox'][2]/w)
                height.append(ann['bbox'][3]/h)
                category.append(ann['category_id'])
                image_id.append(id)

        df = pd.DataFrame()
        df['xmin'] = xmin
        df['ymin'] = ymin
        df['xmax'] = xmax
        df['ymax'] = ymax
        df['width'] = width
        df['height'] = height
        df['category'] = category
        df['image_id'] = image_id

        def coco91_to_coco80_class(label):
            x= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
                 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
            return x.index(label)+1

        df['category'] = df['category'].map(lambda x: coco91_to_coco80_class(x))

        df['aspect_ratio'] = df['width']/df['height']
        df['area'] = df['width']*df['height']
        df.to_csv(args.output)  

    

def get_args_parser():
    parser = argparse.ArgumentParser(description='Parse arguments for csv stat creation.')
    
    parser.add_argument('--dset_name', default='lvisv05',type=str, help='Dataset Name lvis|coco|v3det|openimages')
    parser.add_argument(
        '--path', default='../../../datasets/coco/annotations/lvis_v0.5_val.json', help='path to statistics')
    parser.add_argument(
        '--output', default='./output.csv', help='output file',type=str)

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
    print('end of program')