from where import get
import os

os.makedirs('run_out', exist_ok=True)

for line in open('stat_out').readlines():
	color = line.split()[0][1:]
	print('processing', color)
	filename = get(color)
	os.system(f'python mask.py -f dataset/{filename} -c {color} -s run_out/{color}.png')
