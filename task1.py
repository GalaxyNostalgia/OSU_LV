''' Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je pla´cen
po radnom satu. Koristite ugra ¯denu Python metodu input() . Nakon toga izraˇcunajte koliko
je korisnik zaradio i ispišite na ekran. Na kraju prepravite rješenje na naˇcin da ukupni iznos
izraˇcunavate u zasebnoj funkciji naziva total_euro.
Primjer:
Radni sati: 35 h
eura/h: 8.5
Ukupno: 297.5 eura '''


def total_euro():
    return work_hours * pay_per_hour

work_hours = int(input("Unesi radne sate: "))
pay_per_hour = int(input("Unesi EUR po satu: "))
print(total_euro())
