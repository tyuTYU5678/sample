import os
import shutil
from pathlib import Path
from PIL import Image

def get_wh(path):
    with Image.open(path) as img:
        width, height = img.size
        print(width, height)
    return width, height

dir_txt_0 = "/home/hiraga/work/coil/data/img_hammer_head_mask"
dir_txt_1 = "/home/hiraga/work/coil/data/img2_hammer_head_mask"
dir_txt_2 = "/home/hiraga/work/coil/data/img3_hammer_head_mask"
dir_txts = [dir_txt_0, dir_txt_1, dir_txt_2]

dir_img_0 = "/home/hiraga/work/coil/data/img_hammer_head"
dir_img_1 = "/home/hiraga/work/coil/data/img2_hammer_head"
dir_img_2 = "/home/hiraga/work/coil/data/img3_hammer_head"
dir_imgs = [dir_img_0, dir_img_1, dir_img_2]

dir_output_trn_img = "/home/hiraga/work/groundingdino/data/set001/images/train"
dir_output_trn_txt = "/home/hiraga/work/groundingdino/data/set001/labels/train"
dir_output_val_img = "/home/hiraga/work/groundingdino/data/set001/images/val"
dir_output_val_txt = "/home/hiraga/work/groundingdino/data/set001/labels/val"
dir_output_test = "/home/hiraga/work/groundingdino/data/set001/test"

os.makedirs(dir_output_trn_img, exist_ok=True)
os.makedirs(dir_output_trn_txt, exist_ok=True)
os.makedirs(dir_output_val_img, exist_ok=True)
os.makedirs(dir_output_val_txt, exist_ok=True)
os.makedirs(dir_output_test, exist_ok=True)

def get_fn(lst, dir_txt, idx):
    dir_path = Path(dir_txt)
    fns = [p.stem for p in dir_path.glob("??????.txt")]
    print(fns)
    for fn in fns:
        lst.append((fn, idx))

lst_fn = []
    
get_fn(lst_fn, dir_txt_0, 0)
get_fn(lst_fn, dir_txt_1, 1)
get_fn(lst_fn, dir_txt_2, 2)

print(lst_fn)

nn = len(lst_fn)
print('nn:', nn)

for k, fn in enumerate(lst_fn):
    print("%6d/%d" % (k, nn))
    ft = fn[0]
    idx = fn[1]
    fn_img = ft + ".jpg"
    fn_txt = ft + ".txt"
    path_src_img = os.path.join(dir_imgs[idx], fn_img)
    path_src_txt = os.path.join(dir_txts[idx], fn_txt)
    with open(path_src_txt, "r", encoding="utf-8") as f:
        xmin, ymin, xmax, ymax = map(int, f.read().split())
        #print(xmin, ymin, xmax, ymax)
    width, height = get_wh(path_src_img)
    #exit()    

    if (k % 10) <= 6: # train
        path_dst_img = os.path.join(dir_output_trn_img, fn_img)
        path_dst_txt = os.path.join(dir_output_trn_txt, fn_txt)
    elif (k % 10) <= 7: # validation
        path_dst_img = os.path.join(dir_output_val_img, fn_img)
        path_dst_txt = os.path.join(dir_output_val_txt, fn_txt)
    else: # test
        path_dst_img = os.path.join(dir_output_test, fn_img)
        path_dst_txt = None

    shutil.copy2(path_src_img, path_dst_img)
    if path_dst_txt:
        with open(path_dst_txt, "w", encoding="utf-8") as f:
            xc = (0.5 * (xmax + xmin)) / width
            yc = (0.5 * (ymax + ymin)) / height   
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height   
            f.write("0 %.6f %.6f %.6f %.6f\n" % (xc, yc, w, h))
   
