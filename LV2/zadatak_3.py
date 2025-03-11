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
plt.figure()
plt.imshow(bright_img , cmap="gray")
plt.show()

