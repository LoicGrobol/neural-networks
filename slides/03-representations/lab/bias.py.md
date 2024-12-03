---
jupyter:
  jupytext:
    custom_cell_magics: kql
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: cours-nn
    language: python
    name: python3
---

<!-- LTeX: language=fr -->

# TD : analyse de sentiments express (pour l'anglais)

(inspiré par Karën Fort avec son accord, elle-même inspirée par Elia Robyn Lake, avec son accord)

```python
import pathlib
import random

import numpy as np
import spacy

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

## Charger des plongements statiques pour l'anglais (GloVe en l'occurrence)

On va utiliser SpaCy pour ça. D'abord il nous faut un modèle, s'il n'est pas déjà sur votre machine,
décommentez la cellule suivante et exécutez-la ça va prendre quelques minutes.

```python
# spacy.cli.download("en_core_web_lg")
```

Puis le charger

```python
nlp = spacy.load("en_core_web_lg")
```

Et on peut récupérer des vecteurs de mots

```python
c = nlp("The kitty is happy.")
print(c[1])
print(c[1].vector)
```

Chaque texte aura aussi une représentation vectorielle : la moyenne des représentations vectorielles
de ses mots :

```python
c.vector
```

## Charger un lexique polarisé

Il est déjà dans `data/`. Allez voir comment il est structuré, et utilisez-le pour créer un
dictionnaire qui associe à chaque mot `1` s'il est dans la liste des mots dits positifs et `-1` s'il
est dans la liste des mots négatifs. Faites ça dans une fonction.

```python
def load_lexicon(lexicon_path):
    pass  # À vous de coder


lexicon = load_lexicon("data/opinion-lexicon-English")
lexicon
```

## Apprendre une correspondance embedding-classe


À partir des vecteurs et du lexique, créez deux `arrays`, un `vectors` en empilant les
représentations vectorielles des mots du lexique et un `targets` en empilant leurs catégories (`1`
ou `-1` donc). Faites une fonction. Pensez à vérifier `.has_vector` avant d'essayer d'extraire un
vecteur.

```python
def make_dataset(lexicon, spacy_model):
    pass  # À vous de coder

vectors, targets = make_dataset(lexicon, nlp)
vectors.shape  # Devrait être (taille du lexique, nombre de dimension des embeddings)
```

Maintenant on peut créer nos jeux de train et test.

```python
train_vectors, test_vectors, train_targets, test_targets = train_test_split(vectors, targets, test_size=0.1, random_state=0)
```

Et entraîner le classifieur (128 itérations). 

```python
model = SGDClassifier(loss="log_loss", random_state=0, max_iter=128)
model.fit(train_vectors, train_targets)
```

## Testons l'exactitude de ce modèle


```python
accuracy_score(model.predict(test_vectors), test_targets)
```

Définissons maintenant une fonction qui permet de visualiser la polarité que ce classifieur prédit
pour certains mots, puis appliquons-le sur des données du test set. L'idée est la suivante : pour
établir la polarité d'un mot, qu'il soit dans le lexique ou pas, on récupère son embedding, puis on
passe ce dernier dans le modèle qui nous renvoie deux log-vraisemeblances négatives : $-\log(p_{+})$
et $-\log(p_{-})$. En les soustrayant on obtient un score qui représentera la polarité (positive ou
négative) du mot.

```python
def word_to_sentiment(model, spacy_model, word):
   pass  # À vous de coder


# Show 16 examples from the test set
for word, label in random.choices(list(lexicon.items()), k=16):
    print(word, label, word_to_sentiment(model, nlp, word), sep="\t")
```

Écrivez la fonction `text_to_sentiment` qui fait la même chose pour tout un texte, en considérant que
la polarité d'un texte c'est juste la somme des polarités de ses mots.

```python
def text_to_sentiment(
    model,
    spacy_model,
    sent,
):
    pass  # À vous de coder
```

```python
text_to_sentiment(model, nlp, "this example is pretty cool")
```

```python
text_to_sentiment(model, nlp, "meh, this example sucks")
```

## Applications

Testez les exemples suivants et proposez une explication. Rappel : si les embeddings ont été
entraînés sur toutes sortes de texte, notre modèle, lui n'a été entraîné que sur un lexique qui ne
contient pas du tout tous les mots utilisés ici.

```python
text_to_sentiment(model, nlp, "My name is John")
```

```python
text_to_sentiment(model, nlp, "My name is Tyrone")
```

```python
text_to_sentiment(model, nlp, "My name is ")
```

```python
text_to_sentiment(model, nlp, "I am Christian")

```

```python
text_to_sentiment(model, nlp, "I am a Christian")
```

```python
text_to_sentiment(model, nlp, "a")
```

```python
text_to_sentiment(model, nlp, "Christian")
```

```python
text_to_sentiment(model, nlp, "Jew")
```

```python
text_to_sentiment(model, nlp, "I am a Jew")
```

```python
text_to_sentiment(model, nlp, "I am a Muslim")
```

```python
text_to_sentiment(model, nlp, "I was born in Germany")
```

```python
text_to_sentiment(model, nlp, "I was born in Yemen")
```

```python
for country in [
    "America",  # Non, c'est pas le nom d'un État.
    "France",
    "Germany",
    "England",  # Non, c'est pas le nom d'un État non plus.
    "Ireland",
    "Ethiopia",
    "Iran",
    "China",
    "Yemen",
    "Japan",
    "Congo",
    "Mexico",
    "Canada",
    "Afghanistan",
    "Laos",
    "Luxembourg",
    "Mauritius",
    "Monaco",
    "Oman",
    "Norway",
    "Romania",
    "Somalia",
]:
    print(country, word_to_sentiment(model, nlp, country), sep="\t")
```

- Est-ce que vous voyez des problèmes ?
- Est-ce que ces problèmes peuvent se régler en changeant le modèle ? Comment ?
- Quel sens ça a de prédire la polarité d'un mot ? Comment on pourrait définir formellement la tâche ?
- Est-ce que vous pouvez imaginer des utilisations problématiques de ce type de modèle ?
