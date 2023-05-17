import os
from PIL import Image
from tqdm import tqdm
import json

DATASET_PATH = 'dataset'
SIZE = 512
BASE_SIZE = 600

def main():
	freq = {}
	file_cnt = len(os.listdir(DATASET_PATH))
	for filename in tqdm(os.listdir(DATASET_PATH)):
		img = Image.open(os.path.join(DATASET_PATH, filename)).crop((BASE_SIZE, 0, BASE_SIZE + SIZE, SIZE)).load()
		for i in range(SIZE):
			for j in range(SIZE):
				clr = img[i, j]
				if clr not in freq:
					freq[clr] = 1
				else:
					freq[clr] += 1
	by_occ = list(sorted(freq.items(), key=lambda pr: pr[1]))
	by_occ.reverse()
	for i in range(20):
		frac = by_occ[i][1] / (SIZE * SIZE * file_cnt)
		print('#%02x%02x%02x' % by_occ[i][0], f'{frac:.2%}')


if __name__ == '__main__':
	main()
