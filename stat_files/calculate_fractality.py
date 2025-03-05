import sys
sys.path.insert(1, '../') 
import numpy as np
import pandas as pd 
from tqdm import tqdm
from multiprocessing import Pool
from scipy import optimize
import argparse


def main(args):
    gt = pd.read_csv(args.path)
    dset_name = args.dset_name
    class_names = []
    global calculate_fract
    if dset_name =='coco':
        num_classes = 80
        with open('./coco_names.txt','r') as file:
            for name in file.readlines():
                class_names.append(name.rstrip())
    elif dset_name =='lvisv1':
        num_classes = 1203
        with open('./lvis_names.txt','r') as file:
            for name in file.readlines():
                class_names.append(name.rstrip())
    elif dset_name =='v3det':
        num_classes = 13204
        with open('/home/k00885418/data/V3DET/annotations/category_name_13204_v3det_2023_v1.txt','r') as file:
            for name in file.readlines():
                class_names.append(name.rstrip())
    elif dset_name =='openimages':
        num_classes = 500
        class_names = pd.read_csv('/home/k00885418/data/OpenImages/challenge2019/cls-label-description.csv', names = ['LabelName', 'Label', 'category'])
        class_names = class_names.sort_values('category')
        class_names = class_names['Label'].tolist()
    else:
        num_classes = 1230
        with open('./lvis_names_v0.5.txt','r') as file:
            for name in file.readlines():
                class_names.append(name.rstrip())
        
    cx=np.array(gt['xmin'])+np.array(gt['width'])/2
    cy = np.array(gt['ymin'])+np.array(gt['height'])/2
    gt_categories=np.array(gt['category'])-1
    dimensions = np.arange(args.dims)+1
    frequencies=np.bincount(gt_categories,minlength=num_classes)
    variant = args.variant

    def calculate_fract(dim,cx=cx,cy=cy,gt_categories=gt_categories,num_classes=num_classes,variant=variant):
        frequency=np.bincount(gt_categories,minlength=num_classes)
        step = 1/dim
        gt_loc_bias=np.zeros((dim,dim,num_classes))
        gt_img=np.zeros((dim,dim))

        for j in range(dim):
            for i in range(dim):
                dimx=[i*step,(i+1)*step]
                maskx= (cx>=dimx[0])&(cx<dimx[1])
                dimy=[j*step,(j+1)*step]
                masky= (cy>=dimy[0])&(cy<dimy[1])
                mask_final=maskx&masky
                gt_img[j,i]=mask_final.sum()
                g=gt_categories[mask_final]
                bins=np.bincount(g,minlength=num_classes)
                gt_loc_bias[j,i,:] = bins
        boxes = (gt_loc_bias>0).sum(axis=0).sum(axis=0)
        if variant =='box_dim':
            return boxes
        elif variant =='info_dim':
            return (dim**2)/boxes # 1/p
        elif variant =='smooth_info_dim':
            return (dim**2)/(boxes+1) # 1/p
    
    p = Pool(args.workers)
    grid_size = args.dims
    dims = np.arange(grid_size)+1
    boxes=p.map(calculate_fract, dims)    

    def fit(x, A, Df):
        """
        User defined function for scipy.optimize.curve_fit(),
        which will find optimal values for A and Df.
        """
        return Df * x + A
    
    fractality =  []
    for k in range(num_classes):
        N = [boxes[i][k] for i in np.arange(grid_size)]
        cuttof_ponts = (frequencies[k])>=dims*dims
        if frequencies[k]<4:
            fractality.append(1)
        else:
            if variant =='smooth_info_dim':
                popt, pcov =optimize.curve_fit(fit, np.log(dims)[cuttof_ponts], np.log(N)[cuttof_ponts]+1, maxfev=100000)
            else:
                popt, pcov =optimize.curve_fit(fit, np.log(dims)[cuttof_ponts], np.log(N)[cuttof_ponts], maxfev=100000)
            if variant =='box_dim':
                if popt[1]<1:
                    fractality.append(1)
                else:
                    fractality.append(popt[1])
            else:
                fractality.append(popt[1])
                
    df = pd.DataFrame()
    df['class_names'] = class_names
    df['fractal_dimension'] = fractality
    
    df.to_csv(args.output)
            


def get_args_parser():
    parser = argparse.ArgumentParser(description='Parse arguments for per shot acc.')
    
    parser.add_argument('--dset_name', default='lvisv1',type=str, help='Dataset Name lvis|coco')
    parser.add_argument(
        '--path', default='./lvisv1_train_stats.csv', help='path to statistics')
    parser.add_argument(
        '--output', default='./fractal_dims_lvisv1_train.csv', help='output file',type=str)
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',help='number of workers (default: 32)')
    parser.add_argument('-d', '--dims', default=64, type=int, metavar='N',help='grid max dimension')
    parser.add_argument('--variant', default='box_dim',help='choose between box_dim, info_dim')
    

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
    print('end of program')