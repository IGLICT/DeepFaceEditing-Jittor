import jittor as jt
from jittor import Module
from jittor import nn
import numpy as np
import jittor.transform as transform
from PIL import Image
from combine_model import Combine_Model
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
    parser.add_argument("--geo", type=str, default = "./images/geometry.png", help = "the path of geometry image")
    parser.add_argument("--appear", type=str, default = "./images/appearance.png", help = "the path of appearance image")
    parser.add_argument("--output", type=str, default = "./results/sketch_result.png", help = "the path of output image")
    parser.add_argument("--cuda", type=int, default = 1, help = "use cuda or cpu: 0 , cpu; 1 , gpu")
    parser.add_argument("--geo_type", type=str, default="sketch", help = "extract geometry from image or sketch: sketch / image")
    parser.add_argument("--gen_sketch", action='store_true', help = "with --gen_sketch, extract sketch from real image")
    args = parser.parse_args()

    jt.flags.use_cuda = args.cuda

    if args.gen_sketch:
        sketch_netG = networks.GlobalGenerator(input_nc = 3, output_nc = 3, 
                                        ngf = 32, n_downsampling = 4, n_blocks = 9)
        print(sketch_netG)
        Part_gen_dict = sketch_netG.state_dict()
        for k,v in Part_gen_dict.items():
            print(k)
        sketch_netG.load("./checkpoints/sketch_generator.pkl")
        geo_img = read_img(args.geo)
        with jt.no_grad():
            sketch = sketch_netG(geo_img)
            save_img(sketch, args.output)
    else:
        geo_img = read_img(args.geo)
        appear_img = read_img(args.appear)
        model = Combine_Model()
        model.initialize()
        geo_type = args.geo_type
        image_swap = model.inference(geo_img, appear_img, geo_type)
        save_img(image_swap, args.output)


