import numpy as np
import matplotlib.pyplot as plt

white1 = np.ones((50,50))
white2 = np.ones((50,50))
black1 = np.zeros((50,50))
black2 = np.zeros((50,50))

part1 = np.hstack((black1,white1))
part2 = np.hstack((white2,black2))
image = np.vstack((part1,part2))

plt.imshow(image, cmap="gray")
plt.show()