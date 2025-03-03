'''
Napišite program koji od korisnika zahtijeva upis jednog broja koji predstavlja
nekakvu ocjenu i nalazi se izme ¯du 0.0 i 1.0. Ispišite kojoj kategoriji pripada ocjena na temelju
sljede´cih uvjeta:
>= 0.9 A
>= 0.8 B
>= 0.7 C
>= 0.6 D
< 0.6 F
Ako korisnik nije utipkao broj, ispišite na ekran poruku o grešci (koristite try i except naredbe).
Tako ¯der, ako je broj izvan intervala [0.0 i 1.0] potrebno je ispisati odgovaraju´cu poruku.
'''


try:
    grade = float(input("upiši ocjenu između 0.0 i 1.0: "))
    if(1.0 < grade or grade < 0.0):
        print("Broj nije u intervalu")
    elif(1.0 > grade >= 0.9):
        print("A")
    elif(0.9 > grade >= 0.8):
        print("B")
    elif(0.8 > grade >= 0.7):
        print("C")
    elif(0.7 > grade >= 0.6):
        print("D")
    elif(0.6 > grade > 0.0):
        print("F")
except ValueError:
    print("Nije broj")
