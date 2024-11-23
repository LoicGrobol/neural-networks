---
jupyter:
  jupytext:
    custom_cell_magics: kql
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: cours-ml
    language: python
    name: python3
---

<!-- LTeX: language=fr -->
<!-- #region slideshow={"slide_type": "slide"} -->
Le perceptron simple : solutions
===============================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

<!-- #endregion -->

```python
import numpy as np
import matplotlib.pyplot as plt
```

## Principe et historique

```python
def perceptron(inpt, weight):
    """Calcule la sortie du perceptron dont les poids sont `weights` pour l'entrée `inpt`

    Entrées :

    - `inpt` un tableau numpy de dimension $n$
    - `weights` un tableau numpy de dimention $n+1$

    Sortie: un tableau numpy de type booléen et de dimensions $0$
    """
    return np.greater(np.inner(weight, np.concatenate([[1.0], inpt])), 0.0).astype(np.int64)
```

### 🙅🏻 Exo 🙅🏻

> Déterminer la structure et les poids à utiliser pour implémenter une porte OU et une porte NON
> avec des perceptrons simples.

```python
or_weights = np.array([-0.5, 1, 1])
print("x\ty\tx OU y")
for x_i in [0, 1]:
    for y_i in [0, 1]:
        out = perceptron([x_i, y_i], or_weights)
        print(f"{x_i}\t{y_i}\t{out}")
```

```python
not_weights = np.array([1, -1])
print("x\tNON x")
for x_i in [0, 1]:
    out = perceptron([x_i], not_weights)
    print(f"{x_i}\t{out}")
```

## Algorithme du perceptron

### 🎲 Exo 🎲

> <small>Tiré du [cours de François
> Denis](https://pageperso.lis-lab.fr/~francois.denis/IAAM1/TP3_Perceptron.pdf)</small>
>
> La fonction `random_nice_dataset` du script [`perceptron_data.py`](perceptron_data.py) permet de
> générer un jeu de données aléatoire linéairement séparable en deux dimensions.

```python
import perceptron_data

perceptron_data.random_nice_dataset(4)
```

> 1\. À l'aide de cette fonction, générer un jeu de données d'entraînement et un jeu de données de
> test (par exemple de tailles respectivement 128 et 64).

```python
train_set = perceptron_data.random_nice_dataset(128)
test_set = perceptron_data.random_nice_dataset(64)
```

```python
X = np.stack([x for x, _ in train_set])
c = np.array([c for _, c in train_set])

fig = plt.figure(dpi=200)
heatmap = plt.scatter(X[:, 0], X[:, 1], c=c)
plt.show()

```

> 2\. Appliquer l'algorithme du perceptron sur pour apprendre les données de d'entraînement (sans
> utiliser de terme de biais) et tester les performances du classifieur ainsi appris sur les données
> de test.

```python
alpha = 0.1
w = np.array([0.0, 0.0, 0.0])
err = True
while err:
    err = False
    for x, y in train_set:
        y_hat = perceptron(x, w)
        print(y, y_hat)
        # Only do something if the perceptron made an erro
        if y_hat != y:
            err = True
            w = w + (1 - 2 * y) * np.concatenate([[1.0], x])
            print(f"{w=}")
```

> 3\. Représenter (par exemple avec `matplotlib.pyplot`) les données de test (en utilisant des
> couleurs différentes pour les deux classes) ainsi que la frontière du classifieur.



### ➕ Exo ➕

Le jeu de données `perceptron_data.bias` représente un problème de classification à une dimension,
mais pour lequel un terme de biais est nécessaire.

1\. Appliquer votre implémentation précédente de l'algorithme du perceptron (pour un nombre grand,
mais fixé) d'epochs pour constater la non-convergence (par exemple en affichant les poids et
l'erreur de classification moyenne à chaque étape).

2\. Modifier votre implémentation pour introduire un terme de biais et montrer que dans ce cas
l'apprentissage converge.

## Perceptron multi-classe

### 🌷 Exo 🌷

Appliquer l'algorithme du perceptron multiclasse au jeu de données `perceptron_data.iris` et tester
ses performances avec une validation croisée à 8 plis.
