'''
Napišite Python skriptu koja ´ce uˇcitati tekstualnu datoteku naziva song.txt .
Potrebno je napraviti rjeˇcnik koji kao kljuˇceve koristi sve razliˇcite rijeˇci koje se pojavljuju u
datoteci, dok su vrijednosti jednake broju puta koliko se svaka rijeˇc (kljuˇc) pojavljuje u datoteci.
Koliko je rijeˇci koje se pojavljuju samo jednom u datoteci? Ispišite ih.
'''

dict = {}
fhand = open("song.txt")
for line in fhand:
    line = line.rstrip()
    print(line)
    words = line.split()
    for word in words:
        word = word.strip(",")
        if dict.get(word) == None:
            dict[word] = 1
        else:
            dict[word] += 1

fhand.close()

print(dict)

song_once = [key for key, value in dict.items() if value == 1]
print(f"Jedinstveni: {len(song_once)}")
