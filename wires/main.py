import numpy as mp
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import binary_opening

image = np.load(r"D:\HW\wires/wires6.npy")
struct = np.ones((3, 1))

labeled_image = label(image)

for i in range(1, np.max(labeled_image)+1):
    print(f"Original {i}")
    process = binary_opening(labeled_image == i, struct)
    labeled_process = label(process)
    print(f"Processed {np.max(labeled_process)}")

process = binary_opening(image, struct)

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(binary_opening(image, struct))
plt.show()