'''
 Skripta zadatak_1.py generira umjetni binarni klasifikacijski problem s dvije
 ulazne veliˇ cine. Podaci su podijeljeni na skup za uˇ cenje i skup za testiranje modela.
 a) Prikažite podatke za uˇ cenje u x1−x2 ravnini matplotlib biblioteke pri ˇ cemu podatke obojite
 s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
 marker (npr. ’x’). Koristite funkciju scatter koja osim podataka prima i parametre c i
 cmap kojima je mogu´ ce definirati boju svake klase.
 b) Izgradite model logistiˇ cke regresije pomo´ cu scikit-learn biblioteke na temelju skupa poda
taka za uˇ cenje.
 c) Prona¯ dite u atributima izgra¯ denog modela parametre modela. Prikažite granicu odluke
 nauˇcenog modela u ravnini x1 −x2 zajedno s podacima za uˇcenje. Napomena: granica
 odluke u ravnini x1−x2 definirana je kao krivulja: θ0+θ1x1+θ2x2 = 0.
 d) Provedite klasifikaciju skupa podataka za testiranje pomo´ cu izgra¯ denog modela logistiˇ cke
 regresije. Izraˇ cunajte i prikažite matricu zabune na testnim podacima. Izraˇ cunate toˇ cnost,
 preciznost i odziv na skupu podataka za testiranje.
 e) Prikažite skup za testiranje u ravnini x1 −x2. Zelenom bojom oznaˇ cite dobro klasificirane
 primjere dok pogrešno klasificirane primjere oznaˇ cite crnom bojom.
'''


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# a)
plt.scatter(X_train[:,0], X_train[:,1] ,c=y_train, label="Train")
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, label="Test", marker="x")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()

# b)
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

# c)
theta0, theta1, theta2 = LogRegression_model.intercept_[0], LogRegression_model.coef_[0, 0], LogRegression_model.coef_[0, 1] 

x1_vals = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
x2_vals = - (theta0 + theta1 * x1_vals) / theta2
plt.scatter(X_train[:,0], X_train[:,1] ,c=y_train, label="Train")
plt.plot(x1_vals, x2_vals, 'k-', label="Granica odluke")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()

# d)
y_true = y_test
y_pred = LogRegression_model.predict(X_test)
print("Tocnost: ", accuracy_score(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)
print("Matrica zabune: ", cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
disp.plot()
plt.show()
print(classification_report(y_true, y_pred))

# e)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, label="Test", marker="x")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()