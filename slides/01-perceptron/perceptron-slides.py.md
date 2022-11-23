---
jupyter:
  jupytext:
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- LTeX: language=fr -->
<!-- #region slideshow={"slide_type": "slide"} -->
Cours 01: Le perceptron simple
==============================

**Loïc Grobol** [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)

<!-- #endregion -->

```python
from IPython.display import display, Markdown
```

```python
import numpy as np
import matplotlib.pyplot as plt
```

## Principe et historique

[![Schéma d'un neurone avec des légendes pour les organelles et les connexions importantes pour la
communication entre
neurones.](https://upload.wikimedia.org/wikipedia/commons/1/10/Blausen_0657_MultipolarNeuron.png)](https://commons.wikimedia.org/w/index.php?curid=28761830)

Un modèle de neurone biologique (plutôt sensoriel) : une unité qui reçoit plusieurs entrées $x_i$
scalaires (des nombres quoi), en calcule une somme pondérée $z$ (avec des poids $w_i$ prédéfinis) et
renvoie une sortie binaire $y$ ($1$ si $z$ est positif, $0$ sinon).


Autrement dit

$$\begin{align}
z &= \sum_i w_ix_i = w_1 x_1 + w_2 x_2 + … + w_n x_n\\
y &=
    \begin{cases}
        1 & \text{si $z > 0$}\\
        0 & \text{sinon}
    \end{cases}
\end{align}$$

Formulé célèbrement par McCulloch et Pitts (1943) avec des notations différentes

**Attention** selon les auteurices, le cas $z=0$ est traité différemment, pour *Speech and Language
Processing*, par exemple, on renvoie $0$ dans ce cas, c'est donc la convention qu'on suivra, mais
vérifiez à chaque fois.


On peut ajouter un terme de *biais* en fixant $x_0=1$ et $w_0=b$, ce qui donne

$$\begin{equation}
    z = \sum_{i=0}^n w_ix_i = \sum_{i=1}^n w_ix_i + b
\end{equation}$$

Ou schématiquement


![](figures/perceptron/perceptron.svg)


Ou avec du code

```python
def perceptron(inpt, weights):
    """Calcule la sortie du perceptron dont les poids sont `weights` pour l'entrée `inpt`
    
    Entrées :
    
    - `inpt` un tableau numpy de dimension $n$
    - `weights` un tableau numpy de dimention $n+1$
    
    Sortie: un tableau numpy de type booléen et de dimensions $0$
    """
    return (np.inner(weights[1:], inpt) + weights[0]) > 0
```

Implémenté comme une machine, le perceptron Mark I, par Rosenblatt (1958) :

[![Une photographie en noir et blanc d'une machine ressemblant à une grande armoire pleine de fils
électriques](https://upload.wikimedia.org/wikipedia/en/5/52/Mark_I_perceptron.jpeg)](https://en.wikipedia.org/wiki/File:Mark_I_perceptron.jpeg)


**Est-ce que ça vous rappelle quelque chose ?**


🤔


C'est un **classifieur linéaire** dont on a déjà parlé dans le cours précédent.


Les ambitions initiales étaient grandes

> *the embryo of an electronic computer that [the Navy] expects will be able to walk, talk, see, write, reproduce itself and be conscious of its existence.*  
> New York Times, rapporté par Olazaran (1996)


C'est par exemple assez facile de construire un qui réalise l'opération logique élémentaire $\operatorname{ET}$ :

```python
and_weights = np.array([-0.6, 0.5, 0.5])
print("x\ty\tx ET y")
for x_i in [0, 1]:
    for y_i in [0, 1]:
        out = perceptron([x_i, y_i], and_weights).astype(int)
        print(f"{x_i}\t{y_i}\t{out}")
```

Ça marche bien parce que c'est un problème **linéairement séparable** : si on représente $x$ et $y$
dans le plan, on peut tracer une droite qui sépare la parties où $x\operatorname{ET}y$ vaut $1$ et
la partie où ça vaut $0$ :

```python
import tol_colors as tc

x = np.array([0, 1])
y = np.array([0, 1])
X, Y = np.meshgrid(x, y)
Z = np.logical_and(X, Y)

fig = plt.figure(dpi=200)

heatmap = plt.scatter(X, Y, c=Z, cmap=tc.tol_cmap("sunset"))
plt.colorbar(heatmap)
plt.show()
```

Ici voilà les valeurs que renvoie notre neurone :

```python
import tol_colors as tc

x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)
X, Y = np.meshgrid(x, y)
Z = 0.5*X + 0.5*Y - 0.6 > 0

fig = plt.figure(dpi=200)

heatmap = plt.pcolormesh(X, Y, Z, shading="auto", cmap=tc.tol_cmap("sunset"))
plt.colorbar(heatmap)
plt.show()
```

On confirme : ça marche !


Ça marche aussi très bien pour $\operatorname{OU}$ et $\operatorname{NON}$

## 🙅🏻 Exo 🙅🏻

Déterminer la structure et les poids à utiliser pour implémenter une porte OU et une porte NON avec
des perceptrons simples.

<!-- ```python
or_weights = np.array([-0.5, 1, 1])
print("x\ty\tx OU y")
for x_i in [0, 1]:
    for y_i in [0, 1]:
        out = perceptron([x_i, y_i], or_weights).astype(int)
        print(f"{x_i}\t{y_i}\t{out}")
```

```python
not_weights = np.array([1, -1])
print("x\tNON x")
for x_i in [0, 1]:
    out = perceptron([x_i], not_weights).astype(int)
    print(f"{x_i}\t{out}")
``` -->

## Algorithme du perceptron

### 🎲 Exo 🎲

<small>Tiré du [cours de François
Denis](https://pageperso.lis-lab.fr/~francois.denis/IAAM1/TP3_Perceptron.pdf)</small>

La fonction `random_nice_dataset` du script [`perceptron_data.py`](perceptron_data.py) permet de
générer un jeu de données aléatoire linéairement séparable en deux dimensions.

```python
import perceptron_data

perceptron_data.random_nice_dataset(4)
```

1\. À l'aide de cette fonction, générer un jeu de données d'entraînement et un jeu de données de
test (par exemple de tailles respectivement 128 et 64).

2\. Appliquer l'algorithme du perceptron sur pour apprendre les données de d'entraînement (sans
utiliser de terme de biais) et tester les performances du classifieur ainsi appris sur les données
de test.

3\. Représenter (par exemple avec `matplotlib.pyplot`) les données de test (en utilisant des
couleurs différentes pour les deux classes) ainsi que la frontière du classifieur.

### ➕ Exo ➕

Le jeu de données `perceptron_data.bias` représente un problème de classification à une dimension,
mais pour lequel un terme de biais est nécessaire.

1\. Appliquer votre implémentation précédente de l'algorithme du perceptron (pour un nombre grand
mais fixé d'epochs) pour constater la non-convergence (par exemple en affichant les poids et
l'erreur de classification moyenne à chaque étape).

2\. Modifier votre implémentation pour introduire un terme de biais et montrer que dans ce cas
l'apprentissage converge.

## Perceptron multi-classe

### 🌷 Exo 🌷

Appliquer l'algorithme du perceptron multiclasse au jeu de données `perceptron_data.iris` et tester
ses performances avec une validation croisée à 8 plis.
