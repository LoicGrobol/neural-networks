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
Le perceptron simple : solutions
===============================

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

### 🙅🏻 Exo 🙅🏻

> Déterminer la structure et les poids à utiliser pour implémenter une porte OU et une porte NON
> avec des perceptrons simples.

```python
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
```

## Algorithme du perceptron

L'algorithme du perceptron de Rosenblatt permet de trouver des poids pour lesquels le perceptron
simple partitionne de façon exacte un jeu de données avec un étiquetage linéairement séparable.

On va supposer ici pour simplifier les notations qu'on utilise comme classe $-1$ et $1$ au lieu de
$0$ et $1$ et on considère un taux d'apprentissage $α>0$.

Alors l'algorithme prend la forme suivante :

- Initialiser le vecteur de poids $W$ à des valeurs arbitraires.
- Tant qu'il reste des points $X$ mal classifiés.

  - Pour chaque couple $(X, y) \in \mathcal{D}$ :

    - Calculer $z = \langle W | X \rangle$.
    - Si $y×z ≤ 0$:
      - $W←W+α×y×X$

Notez que :

- La condition $y×z ≤ 0$ est une façon compressée de dire “si $y$ et $z$ sont de même signe” et donc
  “si $\hat{y}= y”.
- La mise à jour de $W$ va tirer $z$ dans la direction de $y$ : calculer $\langle W + αyX | X
  \rangle$ pour s'en convaincre.
- On peut compresser la condition et la mise à jour en une seule ligne : $W←W+α(y-\hat{y})X$.

Sous réserve que le jeu de données soit effectivement linéairement séparable, l'algorithme termine
toujours (et on peut même estimer sa vitesse de convergence), un résultat parfois appelé *théorème
de convergence de Novikov*.

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

1\. Appliquer votre implémentation précédente de l'algorithme du perceptron (pour un nombre grand,
mais fixé) d'epochs pour constater la non-convergence (par exemple en affichant les poids et
l'erreur de classification moyenne à chaque étape).

2\. Modifier votre implémentation pour introduire un terme de biais et montrer que dans ce cas
l'apprentissage converge.

## Perceptron multi-classe

Le cas d'un problème de classification à $n$ classe se traite en prédisant un score par classe avec
une fonction linéaire par classe et en affectant la classe pour laquelle le score est maximal.
Formellement on dispose donc d'un $n$-uplet de poids $W_1, …, W_n$ et on a pour un exemple $X$ :

$$\begin{align}
    z_1 &= \langle W_1 | X \rangle\\
        &⋮\\
    z_n &= \langle W_n | X \rangle
    \hat{y} &= \argmax_k z_k
\end{align}$$

Moralement on peut y penser comme avoir $n$ perceptrons, un par classe.

L'algorithme d'apprentissage du perceptron simple simple s'adapte simplement : pour les exemples mal
classifiés on ajuste les $W_k$ de façon à ce que le score $z_y$ de la classe correcte augmente et à
ce que le score de la classe prédite diminue. En pseudo-code :

- Tant qu'on est pas satisfait⋅e:
  - Pour chaque $(X, y)∈\mathcal{D}$:
    - Pour chaque $k∈⟦1, n⟧$:
      - Calculer $z_k=\langle W_k | X \rangle$
    - Déterminer $\hat{y}=\argmax_k z_k$
    - Si $\hat{y}\neq y$:
      - $W_y←W_y+αX$
      - $W_{\hat{y}}←W_{\hat{y}}-αX$

Le critère de satisfaction peut être comme précédemment “jusqu'à ce qu'il n'y ait plus d'erreur”,
mais comme précédemment ce n'est atteignable qu'avec des conditions contraignantes sur les données.
Un critère d'arrêt plus réaliste peut être un nombre maximal d'étapes ou l'arrêt de l'amélioration
des performances.

**Note** : calculer les $n$ produits scalaires $\langle W_k | X \rangle$ revient à multiplier $X$ à
gauche par la matrice dont les colonnes sont les $W_k$. Autrement dit si on note $W_k = (w_{ℓ, k})$
et qu'on pose

$$
W =
    \begin{pmatrix}
        w_{1,1} & \ldots & w_{1,k}\\
        \vdots  & \ddots & \vdots\\
        w_{1,n} & \ldots & w_{n,k}\\
    \end{pmatrix}
$$

Alors on a

$$
Z = \begin{pmatrix}z_1\\\vdots\\z_n\end{pmatrix} = W×X
$$

Le calcul de ce produit matriciel étant beaucoup plus rapide sur machine que l'écriture d'une
boucle, il est fortement recommandé de l'utiliser, l'algorithme devenant alors :

- Tant qu'on est pas satisfait⋅e:
  - Pour chaque $(X, y)∈\mathcal{D}$:
    - Calculer $Z=W×X$
    - Déterminer $\hat{y}=\argmax_k z_k$
    - Si $\hat{y}\neq y$:
      - $W_y←W_y+αX$
      - $W_{\hat{y}}←W_{\hat{y}}-αX$

### 🌷 Exo 🌷

Appliquer l'algorithme du perceptron multiclasse au jeu de données `perceptron_data.iris` et tester
ses performances avec une validation croisée à 8 plis.
