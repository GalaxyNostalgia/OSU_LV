'''
 Kvantizacija boje je proces smanjivanja broja razliˇ citih boja u digitalnoj slici, ali
 uzimaju´ci u obzir da rezultantna slika vizualno bude što sliˇcnija originalnoj slici. Jednostavan
 naˇcin kvantizacije boje može se posti´ci primjenom algoritma K srednjih vrijednosti na RGB
 vrijednosti elemenata originalne slike. Kvantizacija se tada postiže zamjenom vrijednosti svakog
 elementa originalne slike s njemu najbližim centrom. Na slici 7.3a dan je primjer originalne
 slike koja sadrži ukupno 106,276 boja, dok je na slici 7.3b prikazana rezultantna slika nakon
 kvantizacije i koja sadrži samo 5 boja koje su odre¯ dene algoritmom K srednjih vrijednosti.
 1. Otvorite skriptu zadatak_2.py. Ova skripta uˇcitava originalnu RGB sliku test_1.jpg
 te ju transformira u podatkovni skup koji dimenzijama odgovara izrazu (7.2) pri ˇ cemu je n
 broj elemenata slike, a m je jednak 3. Koliko je razliˇ citih boja prisutno u ovoj slici?
 2. Primijenite algoritam K srednjih vrijednosti koji ´ce prona´ci grupe u RGB vrijednostima
 elemenata originalne slike.
 3. Vrijednost svakog elementa slike originalne slike zamijeni s njemu pripadaju´ cim centrom.
 4. Usporedite dobivenu sliku s originalnom. Mijenjate broj grupa K. Komentirajte dobivene
 rezultate.
 5. Primijenite postupak i na ostale dostupne slike.
 6. Grafiˇcki prikažite ovisnost J o broju grupa K. Koristite atribut inertia objekta klase
 KMeans. Možete li uoˇ citi lakat koji upu´ cuje na optimalni broj grupa?
 7. Elemente slike koji pripadaju jednoj grupi prikažite kao zasebnu binarnu sliku. Što
 primje´ cujete?
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle

# ucitaj sliku
img1 = Image.imread("imgs\\test_1.jpg")
img2 = Image.imread("imgs\\test_2.jpg")
img3 = Image.imread("imgs\\test_3.jpg")
img4 = Image.imread("imgs\\test_4.jpg")
img5 = Image.imread("imgs\\test_5.jpg")
img6 = Image.imread("imgs\\test_6.jpg")

# pretvori vrijednosti elemenata slike u raspon 0 do 1
prvaimg = img1.astype(np.float64) / 255
drugaimg = img2.astype(np.float64) / 255
trecaimg = img3.astype(np.float64) / 255
cetvrtaimg = img4.astype(np.float64)
petaimg = img5.astype(np.float64) / 255
sestaimg = img6.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w1,h1,d1 = prvaimg.shape
img_array1 = np.reshape(prvaimg, (w1*h1, d1))

w2,h2,d2 = drugaimg.shape
img_array2 = np.reshape(drugaimg, (w2*h2, d2))

w3,h3,d3 = trecaimg.shape
img_array3 = np.reshape(trecaimg, (w3*h3, d3))

w4,h4,d4 = cetvrtaimg.shape
img_array4 = np.reshape(cetvrtaimg, (w4*h4, d4))

w5,h5,d5 = petaimg.shape
img_array5 = np.reshape(petaimg, (w5*h5, d5))

w6,h6,d6 = sestaimg.shape
img_array6 = np.reshape(sestaimg, (w6*h6, d6))

image_array_sample1 = shuffle(img_array1, random_state=0, n_samples=1_000)
kmeans1 = KMeans(n_clusters=5, random_state=0).fit(image_array_sample1)

image_array_sample2 = shuffle(img_array2, random_state=0, n_samples=1_000)
kmeans2 = KMeans(n_clusters=5, random_state=0).fit(image_array_sample2)

image_array_sample3 = shuffle(img_array3, random_state=0, n_samples=1_000)
kmeans3 = KMeans(n_clusters=5, random_state=0).fit(image_array_sample3)

image_array_sample4 = shuffle(img_array4, random_state=0, n_samples=1_000)
kmeans4 = KMeans(n_clusters=5, random_state=0).fit(image_array_sample4)

image_array_sample5 = shuffle(img_array5, random_state=0, n_samples=1_000)
kmeans5 = KMeans(n_clusters=5, random_state=0).fit(image_array_sample5)

image_array_sample6 = shuffle(img_array6, random_state=0, n_samples=1_000)
kmeans6 = KMeans(n_clusters=5, random_state=0).fit(image_array_sample6)

# Get labels for all points
labels1 = kmeans1.predict(img_array1)
labels2 = kmeans2.predict(img_array2)
labels3 = kmeans3.predict(img_array3)
labels4 = kmeans4.predict(img_array4)
labels5 = kmeans5.predict(img_array5)
labels6 = kmeans6.predict(img_array6)


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)

# Create a 2x6 grid for displaying original and quantized images side by side
fig, axes = plt.subplots(2, 6, figsize=(20, 8))  # 2 rows, 6 columns
axes = axes.ravel()  # Flatten the 2D array of axes for easier indexing

# List of original and quantized images with titles
images_and_titles = [
    (img1, "Original image 1"),
    (recreate_image(kmeans1.cluster_centers_, labels1, w1, h1), "Quantized image 1"),
    (img2, "Original image 2"),
    (recreate_image(kmeans2.cluster_centers_, labels2, w2, h2), "Quantized image 2"),
    (img3, "Original image 3"),
    (recreate_image(kmeans3.cluster_centers_, labels3, w3, h3), "Quantized image 3"),
    (img4, "Original image 4"),
    (recreate_image(kmeans4.cluster_centers_, labels4, w4, h4), "Quantized image 4"),
    (img5, "Original image 5"),
    (recreate_image(kmeans5.cluster_centers_, labels5, w5, h5), "Quantized image 5"),
    (img6, "Original image 6"),
    (recreate_image(kmeans6.cluster_centers_, labels6, w6, h6), "Quantized image 6")
]

# Display each image in the subplot
for i, (image, title) in enumerate(images_and_titles):
    axes[i].imshow(image)
    axes[i].set_title(title)
    axes[i].axis("off")  # Turn off axes for better visualization

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(img_array1)
    inertias.append(kmeans.inertia_)

''' kmeans.fit(img_array2)
    inertias.append(kmeans.inertia_)

    kmeans.fit(img_array3)
    inertias.append(kmeans.inertia_)

    kmeans.fit(img_array4)
    inertias.append(kmeans.inertia_)

    kmeans.fit(img_array5)
    inertias.append(kmeans.inertia_)

    kmeans.fit(img_array6)
    inertias.append(kmeans.inertia_)'''

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()