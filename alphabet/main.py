import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

save_path = Path(__file__).parent

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1

def extractor(region):
    cy, cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    perimeter = region.perimeter / region.image.size
    holes = count_holes(region)
    vlines = (np.sum(region.image.shape, 0) == region.image.shape[0]).sum()
    hlines = (np.sum(region.image.shape, 1) == region.image.shape[1]).sum()
    eccentricity = region.eccentricity
    aspect = region.image.shape[0] / region.image.shape[1]

    return np.array([region.area/region.image.size, cy, cx, perimeter, holes, vlines, hlines, eccentricity, aspect])

def classificator(region):
    img = region.image
    h, w = img.shape
    total = h * w
    sum_img = img.sum()

    v_ratio = np.sum(np.sum(img, axis=0) == h) / w
    h_ratio = np.sum(np.sum(img, axis=1) == w) / h
    upper = np.sum(img[:h // 2, :])
    lower = np.sum(img[h // 2:, :])

    holes = count_holes(region)

    if holes == 2:
        return "B" if v_ratio > 0.2 else "8"

    if holes == 1:
        if v_ratio > 0.2:
            return "P" if upper > lower * 1.1 else "D"
        else:
            return "O" if upper / lower > 0.8 else "A"

    #1, W, X, *, /, -
    if sum_img == total:
        return "-"

    aspect = min(h, w) / max(h, w)
    if aspect > 0.9:
        return "*"

    if v_ratio > 0 and h_ratio > 0:
        return "1"

    background = np.logical_not(img)
    labeled_bg = label(background)
    bays = sum(1 for r in regionprops(labeled_bg) if r.area > 3)

    if bays == 2:
        return "/"
    elif bays == 4:
        return "X"
    elif bays == 5:
        return "W"
    else:
        return "?"

image = imread("symbols.png")[:, :, :-1]
abinary = image.mean(2) > 0
alphabet = label(abinary)
aprops = regionprops(alphabet)

result = {}

image_path = save_path / "out3"
image_path.mkdir(exist_ok=True)

plt.figure(figsize=(5, 7))

for region in aprops:
    symbol = classificator(region)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1
    plt.cla()
    plt.title(f"Класс - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")
print(result)
print(f"Процент распознования: {1 - result.get("?", 0) / len(aprops)}")