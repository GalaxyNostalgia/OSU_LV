'''
a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka veliˇcina? Postoje li izostale ili
duplicirane vrijednosti? Obrišite ih ako postoje. Kategoriˇcke veliˇcine konvertirajte u tip
category.
b) Koja tri automobila ima najve´cu odnosno najmanju gradsku potrošnju? Ispišite u terminal:
ime proizvod¯acˇa, model vozila i kolika je gradska potrošnja.
c) Koliko vozila ima veliˇcinu motora izme ¯ du 2.5 i 3.5 L? Kolika je prosjeˇcna C02 emisija
plinova za ova vozila?
d) Koliko mjerenja se odnosi na vozila proizvo ¯ daˇca Audi? Kolika je prosjeˇcna emisija C02
plinova automobila proizvod¯acˇa Audi koji imaju 4 cilindara?
e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjeˇcna emisija C02 plinova s obzirom na
broj cilindara?
f) Kolika je prosjeˇcna gradska potrošnja u sluˇcaju vozila koja koriste dizel, a kolika za vozila
koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najve´cu gradsku potrošnju goriva?
h) Koliko ima vozila ima ruˇcni tip mjenjaˇca (bez obzira na broj brzina)?
i) Izracˇunajte korelaciju izmed¯u numericˇkih velicˇina. Komentirajte dobiveni rezultat.
'''


import pandas as pd
print("Zadatak a")
data = pd.read_csv ('data_C02_emission.csv')

data.dropna(axis = 0)
data.drop_duplicates()
data = data.reset_index(drop = True)

print(f"Sadrži: {len(data)}")

print(f"Info: {data.info()}")

print(data)
print(data.head(5))
print(data.tail(3))
print(data.info())
print(data.describe())
print(data.max())
print(data.min())

print("Zadatak b")
highest_consumption = data.nlargest(3, 'Fuel Consumption City (L/100km)')

lowest_consumption = data.nsmallest(3, 'Fuel Consumption City (L/100km)')

print("Three cars with the highest city fuel consumption:")
for index, row in highest_consumption.iterrows():
    print(f"Manufacturer: {row['Make']}, Model: {row['Model']}, City Fuel Consumption: {row['Fuel Consumption City (L/100km)']} L/100km")

print("\nThree cars with the lowest city fuel consumption:")
for index, row in lowest_consumption.iterrows():
    print(f"Manufacturer: {row['Make']}, Model: {row['Model']}, City Fuel Consumption: {row['Fuel Consumption City (L/100km)']} L/100km")

print("Zadatak c")
engine_size = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
average_c02 = engine_size['CO2 Emissions (g/km)'].mean()
print(f"Number of vehicles with engine size between 2.5 and 3.5 L: {len(engine_size)} and average C02 emissions: {average_c02} g/km")

print("Zadatak d")
audi_vehicles = data[data['Make'] == 'Audi']
print(f"Number of Audi vehicles: {len(audi_vehicles)}")
audi4cylinder = audi_vehicles[audi_vehicles['Cylinders'] == 4]
print(f"Audi 4 cylinder emissions: {audi4cylinder['CO2 Emissions (g/km)'].mean()} g/km")

print("Zadatak e")

print(data['Cylinders'].value_counts())
threecylinder = data[data['Cylinders'] == 3]
fourcylinder = data[data['Cylinders'] == 4]
fivecylinder = data[data['Cylinders'] == 5]
sixcylinder = data[data['Cylinders'] == 6]
eightcylinder = data[data['Cylinders'] == 8]
tencylinder = data[data['Cylinders'] == 10]
twelvecylinder = data[data['Cylinders'] == 12]
sixteencylinder = data[data['Cylinders'] == 16]

print(f"Average C02 emissions for 3 cylinder vehicles: {threecylinder['CO2 Emissions (g/km)'].mean()} g/km")
print(f"Average C02 emissions for 4 cylinder vehicles: {fourcylinder['CO2 Emissions (g/km)'].mean()} g/km")
print(f"Average C02 emissions for 5 cylinder vehicles: {fivecylinder['CO2 Emissions (g/km)'].mean()} g/km")
print(f"Average C02 emissions for 6 cylinder vehicles: {sixcylinder['CO2 Emissions (g/km)'].mean()} g/km")
print(f"Average C02 emissions for 8 cylinder vehicles: {eightcylinder['CO2 Emissions (g/km)'].mean()} g/km")
print(f"Average C02 emissions for 10 cylinder vehicles: {tencylinder['CO2 Emissions (g/km)'].mean()} g/km")
print(f"Average C02 emissions for 12 cylinder vehicles: {twelvecylinder['CO2 Emissions (g/km)'].mean()} g/km")
print(f"Average C02 emissions for 16 cylinder vehicles: {sixteencylinder['CO2 Emissions (g/km)'].mean()} g/km")

print("Zadatak f")
print(data['Fuel Type'].value_counts())
diesel = data[data['Fuel Type'] == 'D']
regular_gasoline = data[data['Fuel Type'] == 'Z']
print(f"Average C02 emissions for diesel vehicles: {diesel['CO2 Emissions (g/km)'].mean()} g/km")
print(f"Average C02 emissions for regular gasoline vehicles: {regular_gasoline['CO2 Emissions (g/km)'].mean()} g/km")
print(f"Median C02 emissions for diesel vehicles: {diesel['CO2 Emissions (g/km)'].median()} g/km")
print(f"Median C02 emissions for regular gasoline vehicles: {regular_gasoline['CO2 Emissions (g/km)'].median()} g/km")

print("Zadatak g")
diesel4cylinder = diesel[(diesel['Cylinders'] == 4) & (diesel['Fuel Type'] == 'D')]
highest_city_consumption = print(f"{diesel4cylinder.nlargest(1, 'Fuel Consumption City (L/100km)')}")

print("Zadatak h")
print(data['Transmission'].value_counts())
manual = data[data['Transmission'].str.startswith('M')]
print(f"Number of vehicles with manual: {len(manual)}")

print("Zadatak i")
print(data.corr(numeric_only = True))