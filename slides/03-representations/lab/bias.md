---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# TD : analyse de sentiments express (pour l'anglais)

(inspiré par Karën Fort avec son accord, elle-même inspirée par Robyn Speer, avec son accord)

```python
import numpy as np
import pandas as pd
import matplotlib
import seaborn
import re
import statsmodels.formula.api

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

## Charger des plongements statiques pour l'anglais (GloVe en l'occurrence)

D'ici : <https://nlp.stanford.edu/data/glove.42B.300d.zip>

Dézipper le fichier dans le répertoire data/

Ca prend un peu de temps à charger…

```python
def load_embeddings(filename):
    """
    Load a DataFrame from the generalized text format used by word2vec, GloVe,
    fastText. The main point where they differ is
    whether there is an initial line with the dimensions of the matrix.
    """
    labels = []
    rows = []
    with open(filename, encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            items = line.rstrip().split(' ')
            if len(items) == 2:
                # This is a header row giving the shape of the matrix
                continue
            labels.append(items[0])
            values = np.array([float(x) for x in items[1:]], 'f')
            rows.append(values)
    
    arr = np.vstack(rows)
    return pd.DataFrame(arr, index=labels, dtype='f')

embeddings = load_embeddings('data/glove.42B.300d.txt')
embeddings.shape
```

## A faire : Charger un lexique polarisé

Ici (premier lien) : https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon

L'enregistrer dans votre répertoire notebook, sous data/.

```python
def load_lexicon(filename):
    """
    Load a file from Bing Liu's sentiment lexicon
    (https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html), containing
    English words in Latin-1 encoding.

    
    One file contains a list of positive words, and the other contains
    a list of negative words. The files contain comment lines starting
    with ';' and blank lines, which should be skipped.
    """
    lexicon = []

    with open(filename, encoding="iso8859-1") as in_stream:
        for row in in_stream:
            if not row or row.isspace() or row.startswith(";"):
                continue
            lexicon.append(row.rstrip())
    return lexicon
https://www.youtube.com/watch?v=tlOIHko8ySg
pos_words = load_lexicon('data/opinion-lexicon-English/positive-words.txt')
neg_words = load_lexicon('data/opinion-lexicon-English/negative-words.txt')
pos_words[:8]
```

Les data points que nous avons ici sont les embeddings de ces mots positifs et négatifs. Nous utilisons la fonction .loc[] de Pandas pour obtenir les embeddings des mots.

Certains mots ne sont pas dans le vocabulaire de GloVe, en particulier les erreurs de type "fancinating". Ces mots vont donc correspondre à des lignes pleines de NaN pour indiquer que les embeddings sont manquants. Nous devons les supprimer avec embeddings.index.intersection(pos_words).

```python
pos_vectors = embeddings.loc[embeddings.index.intersection(pos_words)]#.dropna()
neg_vectors = embeddings.loc[embeddings.index.intersection(neg_words)]#.dropna()
```

Maintenant créons des tableaux des entrées et sorties. Les entrées sont les embeddings et les sorties sont 1 pour mots qualifiés de positifs et -1 pour les négatifs. Nous nous assurons également de garder en mémoire les mots qui correspondent, afin de pouvoir interpréter les résultats.

```python
vectors = pd.concat([pos_vectors, neg_vectors])
targets = np.array([1 for entry in pos_vectors.index] + [-1 for entry in neg_vectors.index])
labels = list(pos_vectors.index) + list(neg_vectors.index)
```

On utilise la fonction scikit-learn train_test_split pour séparer simultanément les vecteurs d'entrée, les valeurs de sortie et les étiquettes (labels) entre données d'entraînement (training) et de test (10 %).

```python
train_vectors, test_vectors, train_targets, test_targets, train_labels, test_labels = train_test_split(vectors, targets, labels, test_size=0.1, random_state=0)
```

Maintenant on entraîne le classifieur (100 itérations). 

```python
model = SGDClassifier(loss='log_loss', random_state=0, max_iter=100)#n_iter
model.fit(train_vectors, train_targets)
```

## Testons l'exactitude de ce modèle


```python
accuracy_score(model.predict(test_vectors), test_targets)
```

Définissons maintenant une fonction qui permet de visualiser la polarité que ce classifieur prédit pour certains mots, puis appliquons-le sur des données du test set.

```python
def vecs_to_sentiment(vecs):
    # predict_log_proba gives the log probability for each class
    predictions = model.predict_log_proba(vecs)

    # To see an overall positive vs. negative classification in one number,
    # we take the log probability of positive sentiment minus the log
    # probability of negative sentiment.
    return predictions[:, 1] - predictions[:, 0]


def words_to_sentiment(words):
    vecs = embeddings.loc[words].dropna()
    log_odds = vecs_to_sentiment(vecs)
    return pd.DataFrame({'sentiment': log_odds}, index=vecs.index)

# Show 20 examples from the test set
words_to_sentiment(test_labels).iloc[:20] #.ix

```

```python

```

```python

```

Nous pouvons maintenant comparer grossièrement les polarités de différentes phrases, avec le code ci-dessous :

```python
import re
TOKEN_RE = re.compile(r"\w.*?\b")
# The regex above finds tokens that start with a word-like character (\w), and continues
# matching characters (.+?) until the next word break (\b). It's a relatively simple
# expression that manages to extract something very much like words from text.


def text_to_sentiment(text):
    tokens = [token.casefold() for token in TOKEN_RE.findall(text)]
    sentiments = words_to_sentiment(tokens)
    return sentiments['sentiment'].mean()
```

```python
text_to_sentiment("this example is pretty cool")
```

```python
text_to_sentiment("meh, this example sucks")
```

```python
text_to_sentiment("my name is john")
```

```python
text_to_sentiment("my name is tyrone")
```

```python
text_to_sentiment("i am christian")
```

```python
text_to_sentiment("i am hebrew")
```
