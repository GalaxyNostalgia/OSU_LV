import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.csv",
                 delimiter=",", dtype=float)
counter = data.shape[0]
print(f"Broj osoba: {counter}")

fig, axs = plt.subplots(1,2, figsize=(12,9), sharey='row')
prvi,drugi = axs

visina = data[:,1]
tezina = data[:,2]

prvi.scatter(tezina, visina)
prvi.set_ylabel("visina")
prvi.set_xlabel("tezina")

fifty = data[::50]
print(fifty)

visinaF = fifty[:,1]
tezinaF = fifty[:,2]

drugi.scatter(tezinaF, visinaF)
drugi.set_ylabel("visina")
drugi.set_xlabel("tezina")

plt.tight_layout()
plt.show()

heights = data[:,1]
print(f"Visina min: {heights.min()}")
print(f"Visina max: {heights.max()}")
print(f"Visina srednja: {heights.mean()}")

muskarci = data[:,0] == 1
zene = data[:,0] == 0

visine_muski = visina[muskarci]
visine_zene = visina[zene]

print(f"Visina muski min: {visine_muski.min()}")
print(f"Visina muski max: {visine_muski.max()}")
print(f"Visina muski srednja: {visine_muski.mean()}")

print(f"Visina zene min: {visine_zene.min()}")
print(f"Visina zene max: {visine_zene.max()}")
print(f"Visina zene srednja: {visine_zene.mean()}")
