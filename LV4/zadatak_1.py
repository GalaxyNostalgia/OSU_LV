'''
Skripta zadatak_1.py ucitava podatkovni skup iz ˇ data_C02_emission.csv.
Potrebno je izgraditi i vrednovati model koji procjenjuje emisiju C02 plinova na temelju ostalih numerickih ulaznih veli ˇ cina. Detalje oko ovog podatkovnog skupa mogu se prona ˇ ci u 3. ´
laboratorijskoj vježbi.
a) Odaberite željene numericke veli ˇ cine speci ˇ ficiranjem liste s nazivima stupaca. Podijelite
podatke na skup za ucenje i skup za testiranje u omjeru 80%-20%. ˇ
b) Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova ´
o jednoj numerickoj veli ˇ cini. Pri tome podatke koji pripadaju skupu za u ˇ cenje ozna ˇ cite ˇ
plavom bojom, a podatke koji pripadaju skupu za testiranje oznacite crvenom bojom. ˇ
c) Izvršite standardizaciju ulaznih velicina skupa za u ˇ cenje. Prikažite histogram vrijednosti ˇ
jedne ulazne velicine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja ˇ
transformirajte ulazne velicine skupa podataka za testiranje. ˇ
d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
povežite ih s izrazom 4.6.
e) Izvršite procjenu izlazne velicine na temelju ulaznih veli ˇ cina skupa za testiranje. Prikažite ˇ
pomocu dijagrama raspršenja odnos izme ´ du stvarnih vrijednosti izlazne veli ¯ cine i procjene ˇ
dobivene modelom.
f) Izvršite vrednovanje modela na nacin da izra ˇ cunate vrijednosti regresijskih metrika na ˇ
skupu podataka za testiranje.
g) Što se dogada s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj ¯
ulaznih velicina?
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm



print("Zadatak a")
data = pd.read_csv('data_C02_emission.csv')
x = data[['Engine Size (L)', 'Fuel Consumption City (L/100km)', 'Cylinders']]
y = data['CO2 Emissions (g/km)']

X_train , X_test , y_train , y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print("Zadatak b")
plt.scatter(X_train['Cylinders'], y_train, c='b')
plt.scatter(X_test['Cylinders'], y_test, c='r')
plt.show()

print("Zadatak c")

sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

X_train_n = pd.DataFrame(X_train_n, columns=['Engine Size (L)', 'Fuel Consumption City (L/100km)', 'Cylinders'])
plt.title("Originalni")
plt.hist(X_train['Cylinders'])
plt.show()
plt.title("Skalirani")
plt.hist(X_train_n['Cylinders'])
plt.show()

print("Zadatak d")

linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(f"Thete: {linearModel.coef_}")
print(f"Presjek: {linearModel.intercept_}")

'''categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
ohe = OneHotEncoder()
one_hot_encoded = ohe.fit_transform(data[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=ohe.get_feature_names_out(categorical_columns))
df_encoded = pd.concat([data, one_hot_df], axis=1)
df_encoded = df_encoded.drop(categorical_columns, axis=1)
print(f"Encoded Employee data : \n{df_encoded}")'
'''
                                 



