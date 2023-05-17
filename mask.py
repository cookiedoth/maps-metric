import argparse
import struct
import codecs
from PIL import Image
import numpy as np

SIZE = 512
BASE_SIZE = 600

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file')
parser.add_argument('-c', '--color')
parser.add_argument('-s', '--save_path', required=False)
args = parser.parse_args()

dst = Image.new('RGB', (2 * SIZE, SIZE))
img1 = Image.open(args.file).crop((BASE_SIZE, 0, BASE_SIZE + SIZE, SIZE))
dst.paste(img1, (0, 0))
arr = np.array(img1)

red, green, blue = bytes.fromhex(args.color)

mask = np.all(arr == np.array([red, green, blue]), axis=-1)
# mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
pic = np.ones((SIZE, SIZE, 3)) * 255
pic[:, :, 1] = np.where(mask, 0, 255)
pic[:, :, 2] = np.where(mask, 0, 255)
pic = pic.astype(np.uint8)
img2 = Image.fromarray(pic, mode='RGB')
dst.paste(img2, (SIZE, 0))
if args.save_path != None:
	dst.save(args.save_path)
else:
	dst.show()
