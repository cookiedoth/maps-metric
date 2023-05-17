import os
from PIL import Image
from tqdm import tqdm
import json
import argparse

DATASET_PATH = 'dataset'
SIZE = 512
BASE_SIZE = 600

def get(color):
	red, green, blue = bytes.fromhex(color)
	target_clr = (red, green, blue)

	freq = {}
	file_cnt = len(os.listdir(DATASET_PATH))
	opt = ''
	opt_cnt = 0
	for filename in tqdm(os.listdir(DATASET_PATH)):
		img = Image.open(os.path.join(DATASET_PATH, filename)).crop((BASE_SIZE, 0, BASE_SIZE + SIZE, SIZE)).load()
		cnt = 0
		for i in range(SIZE):
			for j in range(SIZE):
				clr = img[i, j]
				if (clr == target_clr):
					cnt += 1
		if cnt > opt_cnt:
			opt = filename
			opt_cnt = cnt
	return opt


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--color')
	args = parser.parse_args()
	print(get(args.color))
