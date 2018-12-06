# TweetClassification
Classification de tweets bas ́ee sur le transfert learning

# Utilisation

Lancé 'main.py'

Notre programme ajoute des dimensions aux words embedding basique
de notr Word2Vec se basant sur des base d'analyse d'emotions tels que
Afinn, Depeche Mood, EV etc...
Ensuite notre programme entraine un model en s'entrainant sur 50k
tweet classé en 3 classes afin de finir par l'entrainement d'un
model via 'Transfer learning' sur environs 1500 tweet classé en 7
classes.
