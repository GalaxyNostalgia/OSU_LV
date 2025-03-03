'''
Napišite Python skriptu koja ´ce uˇcitati tekstualnu datoteku naziva SMSSpamCollection.txt.
Ova datoteka sadrži 5574 SMS poruka pri ˇcemu su neke oznaˇcene kao spam, a neke kao ham.
Primjer dijela datoteke:
ham Yup next stop.
ham Ok lar... Joking wif u oni...
spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken’s stuff!
a) Izraˇcunajte koliki je prosjeˇcan broj rijeˇci u SMS porukama koje su tipa ham, a koliko je
prosjeˇcan broj rijeˇci u porukama koje su tipa spam.
b) Koliko SMS poruka koje su tipa spam završava uskliˇcnikom ?
'''


def average():
    ham = 0
    spam = 0
    ham_sum = 0
    spam_sum = 0
    fhand = open("SMSSpamCollection.txt")
    for line in fhand:
        data = line.split("\t")

        if data[0] == "ham":
            ham += 1
            ham_sum += len(data[1].split())
        if data[0] == "spam":
            spam += 1
            spam_sum += len(data[1].split())

    fhand.close()

    ham_avg = ham_sum / ham
    spam_avg = spam_sum / spam

    print(ham_avg)
    print(spam_avg)

def exclamation():
    counter = 0
    fhand = open("SMSSpamCollection.txt")
    for line in fhand:
        data = line.split("\t")
        if data[0] == "spam":
           words = data[1].split()
           if words[-1][-1] == "!":
               counter += 1

    fhand.close()

    return counter

average()
res = exclamation()
print(res)
