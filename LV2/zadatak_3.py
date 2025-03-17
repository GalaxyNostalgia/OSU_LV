'''Skripta zadatak_3.py uˇcitava sliku ’road.jpg’. Manipulacijom odgovaraju ́ce
numpy matrice pokušajte:
a) posvijetliti sliku,
b) prikazati samo drugu ˇcetvrtinu slike po širini,
c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,
d) zrcaliti sliku.'''


import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
img = img[:,:,0].copy()
bright_img = np.clip(img + 50, 0, 255)   
print(img.shape)
print(img.dtype)
plt.figure(1)
plt.imshow(bright_img , cmap="gray")
plt.show()

plt.figure(2)
plt.imshow(img[:,img.shape[1]//2:], cmap="gray")
plt.show()

rotated_img = np.rot90(img)
rotated_img = np.rot90(rotated_img)
rotated_img = np.rot90(rotated_img)
plt.figure(3)
plt.imshow(rotated_img, cmap="gray")
plt.show()

flipped_img = np.fliplr(img)
plt.figure(4)
plt.imshow(flipped_img, cmap="gray")
plt.show()