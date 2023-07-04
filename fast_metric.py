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

def find_closest_classes(image, classes):
    image_4d = image[:, :, np.newaxis, :]
    classes_4d = classes[np.newaxis, np.newaxis, :, :]
    all_dist = np.sum((image_4d - classes_4d) ** 2, axis=3)
    opt = np.argmin(all_dist, axis=2)
    return opt

def read_color_json(path):
    colors = json.loads(open(path).read())
    rgb = np.zeros((len(colors), 3))
    names = []
    i = 0
    for key, value in colors.items():
        rgb[i] = np.array(process_color(key))
        names.append(value)
        i += 1
    return rgb, names

def get_class_matrix(names_pred, names_target):
    result = np.zeros((len(names_pred), len(names_target)))
    for i in range(len(names_pred)):
        for j in range(len(names_target)):
            if names_pred[i] == names_target[j]:
                result[i, j] = 1.0
            elif sorted([names_pred[i], names_target[j]]) == \
                 ['background', 'park']:
                result[i, j] = 0.7
            else:
                result[i, j] = 0.0
    return result

def metric(pred, target, colors_pred, colors_target):
    rgb_pred, names_pred = read_color_json(colors_pred)
    rgb_target, names_target = read_color_json(colors_target)
    pred = np.asarray(pred)
    target = np.asarray(target)
    class_matrix = get_class_matrix(names_pred, names_target)
    pred_classes = find_closest_classes(pred, rgb_pred)
    target_classes = find_closest_classes(target, rgb_target)
    pixel_metric = class_matrix[pred_classes, target_classes]
    mask = np.ones((SIZE, SIZE, 3), dtype=np.uint8) * 255
    mask[pixel_metric == 0] = np.array([255, 0, 0])
    mask[pixel_metric == 1] = np.array([0, 255, 0])
    mask[np.logical_and(pixel_metric > 0, pixel_metric < 1)] = np.array([255, 255, 0])
    return np.sum(pixel_metric) / (SIZE * SIZE), Image.fromarray(mask, mode='RGB')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target')
    parser.add_argument('-p', '--pred')
    args = parser.parse_args()
    pred = Image.open(args.pred)
    target = Image.open(args.target)
    result, mask = metric(pred, target, 'colors2.json', 'colors.json')
    print(f'{result:.2%}')
    mask.show()
