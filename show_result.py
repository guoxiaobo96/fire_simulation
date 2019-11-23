import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import os
import imageio
import shutil
from glob import glob

def vort(x):
    dvdx = x[:,:,1:,1] - x[:,:,:-1,1]
    dudy = x[:,1:,:,0] - x[:,:-1,:,0]
    dvdx = np.concatenate([dvdx,np.expand_dims(dvdx[:,:,-1], axis=2)], axis=2)
    dudy = np.concatenate([dudy,np.expand_dims(dudy[:,-1,:], axis=1)], axis=1)
    return np.expand_dims(dvdx - dudy, axis=-1)


def get_vort_image(x):
    x = vort(x[:,:,:,:2])
    x /= np.abs(x).max() # [-1,1]
    x_img = (x+1)*127.5
    x_img = np.uint8(plt.cm.RdBu(x_img[...,0]/255)*255)[...,:3]
    return x_img

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False, flip=True):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            if flip: grid[h:h+h_width, w:w+w_width] = tensor[k,::-1]
            else: grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False, flip=True, single=False):
    if not single:
        ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                          normalize=normalize, scale_each=scale_each, flip=flip)
    else:
        h, w = tensor.shape[0], tensor.shape[1]
        ndarr = np.zeros([h,w,3], dtype=np.uint8)
        if flip: ndarr = tensor[::-1]
        else: ndarr = tensor
    
    im = Image.fromarray(ndarr)
    im.save(filename)

def build_image_from_file(data_dir, p1, p2):
    target_dir = data_dir + "_" + p1 + "_" + p2
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    data_dir = data_dir + "/" + p1 + "_" + p2 + "_"
    name_list = [str(i) for i in range(200)]
    for name in name_list:
        x = np.load(data_dir + name+ '.npz')["x"]
        if len(x.shape)==3:
            x = np.array([x])
        x = get_vort_image(x)
        save_image(x, target_dir + "/" + name + '.png')
    return target_dir

def build_image_from_tensor(x,data_dir, file_name):
    x = get_vort_image(x/127.5-1)
    save_image(x,data_dir+"/"+str(file_name)+".png")
    
def convert_png2mp4(imgdir, filename, fps, delete_imgdir=False):
    # dirname = os.path.dirname(filename)
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)

    try:
        writer = imageio.get_writer(filename, fps=fps)
    except Exception:
        imageio.plugins.ffmpeg.download()
        writer = imageio.get_writer(filename, fps=fps)
    image_number = [str(i) + ".png" for i in range(len(os.listdir(imgdir)))]
    imgs = [imgdir+"/"+i for i in image_number]
    for img in imgs:
        im = imageio.imread(img)
        writer.append_data(im)
    
    writer.close()
    
    if delete_imgdir: shutil.rmtree(imgdir)

if __name__ == '__main__':
    target_dir = build_image_from_file("./log/smoke_pos21_size5_f200/1119_170410_de_tag/20_2","20","2")
    convert_png2mp4(target_dir,'./video/extend_generated.mp4',30,True)