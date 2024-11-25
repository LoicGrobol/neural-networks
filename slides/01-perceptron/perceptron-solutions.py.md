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


On va utiliser une version vectorisée du perceptron. On peut faire sans (avec des boucles), mais
avec la très puissante fonction
[`np.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html), la vie est douce.

```python
def perceptron(inpt, weights):
    """Calcule la sortie du perceptron dont les poids sont `weights` pour l'entrée `inpt`

    Entrées :

    - `inpt` un tableau numpy de dimension $*×n$
    - `weights` un tableau numpy de dimention $n+1$

    Sortie: un tableau numpy d'entiers de dimension *, tous 0 soit 1.
    """
    inpt = np.array(inpt)
    biased_inpt = (
        np.concatenate(
            (np.full((*inpt.shape[:-1], 1), 1.0), inpt),
            axis=-1,
        )
    )
    return np.greater(
        # En vrai dans ce cas ça marcherait avec `np.dot` mais je veux pas avoir à me souvenir
        # comment elle marche.
        # Ici on dit "le premier argument est de dimension i, le deuxième de dimension (*, i),
        # la sortie doit avoir toutes les dimensions de l'entrée, en enlevant i avec des
        # sommes-produits.
        np.einsum(
            "i,...i->...",
            weights,
            biased_inpt,
        ),
        0.0,
    ).astype(np.int64)
```

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

plt.figure(dpi=200)
plt.scatter(X[:, 0], X[:, 1], c=c, s=2)
plt.show()

```

> 2\. Appliquer l'algorithme du perceptron sur pour apprendre les données de d'entraînement (sans
> utiliser de terme de biais) et tester les performances du classifieur ainsi appris sur les données
> de test.

```python
alpha = 0.1
w = np.array([0.0, 0.0, 0.0])
err = True
n_epoch = 0
while err:
    n_epoch += 1
    print(f"Epoch {n_epoch}")
    err = False
    for x, y in train_set:
        y_hat = perceptron(x, w)
        # Only do something if the perceptron made an erro
        if y_hat != y:
            err = True
            w = w + (2 * y-1) * np.concatenate([[1.0], x])
            print(f"{w=}")
```

> 3\. Représenter (par exemple avec `matplotlib.pyplot`) les données de test (en utilisant des
> couleurs différentes pour les deux classes) ainsi que la frontière du classifieur.



```python
x = np.linspace(-10.0, 10.0, 1000)
y = np.linspace(-10.0, 10.0, 1000)
X, Y = np.meshgrid(x, y)
# On calcule la sortie du perceptron pour chaque point de la grille
# C'est là que pouvoir travailler avec des inputs arbitraires est pratique
Z = perceptron(np.stack((X, Y), axis=-1), w)

X_train = np.stack([x for x, _ in train_set])
c_train = np.array([c for _, c in train_set])

plt.figure(dpi=200)
heatmap = plt.pcolormesh(X, Y, Z, shading="auto")
# Les couleurs sont pas les mêmes mais comme ça on voit les deux classes
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=c_train,
    cmap="seismic",
    s=2,
)
plt.colorbar(heatmap)
plt.show()

```

## Perceptron multi-classe

### 🌷 Exo 🌷

Appliquer l'algorithme du perceptron multiclasse au jeu de données `perceptron_data.iris` et tester
ses performances avec une validation croisée à 8 plis.
