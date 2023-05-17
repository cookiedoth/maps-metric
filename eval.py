from PIL import Image
import numpy as np
import argparse
import json

SIZE = 512

def process_color(color):
	red, green, blue = bytes.fromhex(color)
	return (red, green, blue)

def dist(clr1, clr2):
	return (clr1[0] - clr2[0]) ** 2 + (clr1[1] - clr2[1]) ** 2 + (clr1[2] - clr2[2]) ** 2

def mertic(target, prediction, colors, colors2):
	mask = np.ones((SIZE, SIZE, 3), dtype=np.uint8) * 255
	good = 0
	total = 0
	for i in range(SIZE):
		for j in range(SIZE):
			target_class = 'none'
			prediction_class = ''
			for clr in colors:
				if target[i, j] == clr:
					target_class = colors[clr]
			opt_dist = 10 ** 9
			for clr in colors2:
				d = dist(clr, prediction[i, j])
				if d < opt_dist:
					opt_dist = d
					prediction_class = colors2[clr]
			if i == 418 and j == 0:
				print(prediction[i, j])
				print(i, j, target_class, prediction_class)
			if target_class != 'none':
				total += 1
				if sorted([prediction_class, target_class]) == ['background', 'park']:
					good += 0.7
					mask[j, i] = np.array([255, 255, 0])
				elif prediction_class == target_class:
					good += 1
					mask[j, i] = np.array([0, 255, 0])
				else:
					mask[j, i] = np.array([255, 0, 0])
	return good / total, Image.fromarray(mask, mode='RGB')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--target')
	parser.add_argument('-p', '--prediction')
	args = parser.parse_args()
	colors = json.loads(open('colors.json').read())
	colors = { process_color(key): value for key, value in colors.items() }
	colors2 = json.loads(open('colors2.json').read())
	colors2 = { process_color(key): value for key, value in colors2.items() }
	result, mask = mertic(Image.open(args.target).load(), Image.open(args.prediction).load(), colors, colors2)
	print(f'{result:.2%}')
	mask.show()
