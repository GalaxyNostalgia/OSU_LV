'''
a) Pomo´cu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz.
b) Pomo´cu dijagrama raspršenja prikažite odnos izme ¯ du gradske potrošnje goriva i emisije
C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izme ¯ du
veliˇcina, obojite toˇckice na dijagramu raspršenja s obzirom na tip goriva.
c) Pomo´cu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip
goriva. Primje´cujete li grubu mjernu pogrešku u podacima?
d) Pomo´cu stupˇcastog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu
groupby.
e) Pomo´cu stupˇcastog grafa prikažite na istoj slici prosjeˇcnu C02 emisiju vozila s obzirom na
broj cilindara.
'''
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')
print("Zadatak a")
plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist')
plt.show()

print("Zadatak b")
grouped = data.groupby('Fuel Consumption City (L/100km)')
plt.scatter(data['Fuel Consumption City (L/100km)'], data['CO2 Emissions (g/km)'], c=data['Fuel Type'].astype('category').cat.codes)
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()

print("Zadatak c")
data.boxplot(column='Fuel Consumption City (L/100km)', by='Fuel Type')
plt.show()

print("Zadatak d")
fuel = data.groupby('Fuel Type')
fuel.size().plot(kind='bar')
plt.show()

print("Zadatak e")
cylinders_group = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
plt.figure()
cylinders_group.plot(kind='bar', color='blue')
plt.title('Average CO2 Emissions (g/km) by Number of Cylinders')
plt.xlabel('Number of Cylinders')
plt.ylabel('Average CO2 Emissions (g/km)')
plt.show()
