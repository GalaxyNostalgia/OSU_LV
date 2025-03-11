'''
Napišite program koji od korisnika zahtijeva unos brojeva u beskonaˇcnoj petlji
sve dok korisnik ne upiše „ Done “ (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
(npr. slovo umjesto brojke) na naˇcin da program zanemari taj unos i ispiše odgovaraju´cu poruku.
'''

list = []
infinite = True

while(infinite):
    number = input("Upiši: ")
    if number.isdigit():
        list.append(int(number))
    elif(number == "Done"):
        infinite = False

print(sum(list) / len(list))
print(min(list))
print(max(list))
list.sort()
print(list)
