import jittor as jt
from jittor import Module
from jittor import nn
import numpy as np
import jittor.transform as transform
from PIL import Image
from combine_model import Combine_Model_Projection
import networks
from argparse import ArgumentParser

img_size = 512
transform_image = transform.Compose([
        transform.Resize(size = img_size),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def read_img(path):
    img = Image.open(path).convert('RGB')
    img = transform_image(img)
    img = jt.array(img)
    img = img.unsqueeze(0)
    return img

def save_img(image, path):
    image = image.squeeze(0).detach().numpy()
    image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--geo", type=str, default = "./images/CoarseSketch.jpg", help = "the path of geometry image")
    parser.add_argument("--appear", type=str, default = "./images/29042.jpg", help = "the path of appearance image")
    parser.add_argument("--output", type=str, default = "./results/sketch_gen.png", help = "the path of output image")
    parser.add_argument("--gender", type=int, default = 0, help = "gender of images: 0, female, 1, man")
    parser.add_argument("--cuda", type=int, default = 1, help = "use cuda or cpu: 0 , cpu; 1 , gpu")

    args = parser.parse_args()
    jt.flags.use_cuda = args.cuda

    geo_img = read_img(args.geo)
    appear_img = read_img(args.appear)
    geo_img = geo_img[:,0:1,:,:]

    model = Combine_Model_Projection()
    model.initialize()
    
    gender = args.gender
    part_weights = {'bg': 1.0, 
                  'eye1': 1.0,
                  'eye2': 1.0, 
                  'nose': 1.0, 
                 'mouth': 1.0}
    image_result = model.inference(geo_img, appear_img, gender, part_weights)
    save_img(image_result, args.output)



