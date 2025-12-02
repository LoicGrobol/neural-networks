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
      jupytext_version: 1.18.1
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

Un modèle de neurone biologique (plutôt sensoriel) : une unité qui reçoit plusieurs entrées $x_j$
scalaires (des nombres quoi), en calcule une somme pondérée $z$ (avec des poids $w_j$ prédéfinis) et
renvoie une sortie binaire $y$ ($1$ si $z$ est positif, $0$ sinon).


Autrement dit

$$
\begin{align}
z &= \sum_j w_jx_j = w_1 x_1 + w_2 x_2 + … + w_n x_n\\
\hat{y} &=
    \begin{cases}
        1 & \text{si $z > 0$}\\
        0 & \text{sinon}
    \end{cases}
\end{align}
$$

Formulé célèbrement par McCulloch et Pitts (1943) avec des notations différentes.

Implémenté comme une machine, le perceptron Mark I, par Rosenblatt (1958) :

[![Une photographie en noir et blanc d'une machine ressemblant à une grande armoire pleine de fils
électriques](https://upload.wikimedia.org/wikipedia/en/5/52/Mark_I_perceptron.jpeg)](https://en.wikipedia.org/wiki/File:Mark_I_perceptron.jpeg)

**Attention** selon les auteurices, le cas $z=0$ est traité différemment, pour *Speech and Language
Processing*, par exemple, on renvoie $0$ dans ce cas, c'est donc la convention qu'on suivra, mais
vérifiez à chaque fois.

**Note**: si on note $W$ le vecteur dont les coordonnées sont les $w_j$ et $X$ celui dont les
coordonnés sont les $x_j$, $z$ est le **produit scalaire** de $W$ et $X$, noté $\langle W | X
\rangle$.


On peut ajouter un terme de *biais* en fixant $x_0=1$ et $w_0=b$, ce qui donne

$$
\begin{equation}
    z = \sum_{j=0}^n w_jx_j = \sum_{j=1}^n w_jx_j + b
\end{equation}
$$

Ou schématiquement


![](figures/perceptron/perceptron.svg)


Ou avec du code

```python
def perceptron(inpt, weight):
    z = weight[0] # On commence par le terme de biais
    for i in range(len(inpt)):  # Attention, en Python les indices commencent à 0
        z += inpt[i]*weight[i]
    if z > 0:
        y = 1
    else:
        y = 0
    return y

perceptron([-2.0, -1.0], [-0.5, 0.5])
```

```python
def perceptron(inpt, weight):
    z = weight[0]
    for x, w in zip(inpt, weight[1:]):
        z += w*x
    if z > 0:
        y = 1
    else:
        y = 0
    return y

perceptron([-2.0, -1.0], [1.0, -0.5, 0.5])
```

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



**Est-ce que ça vous rappelle quelque chose ?**


🤔


C'est un **classifieur linéaire** dont on a déjà parlé dans le cours précédent.


Les ambitions initiales étaient grandes

> *the embryo of an electronic computer that [the Navy] expects will be able to walk, talk, see,
> write, reproduce itself and be conscious of its existence.*  
> New York Times, rapporté par Olazaran (1996)


C'est par exemple assez facile de construire un qui réalise l'opération logique élémentaire
$\operatorname{ET}$ :

```python
and_weights = np.array([-0.6, 0.5, 0.5])
print("x\ty\tx ET y")
for x in [0, 1]:
    for y in [0, 1]:
        out = perceptron([x, y], and_weights)
        print(f"{x}\t{y}\t{out}")
```

Ça marche bien parce que c'est un problème **linéairement séparable** : si on représente $x$ et $y$
dans le plan, on peut tracer une droite qui sépare la partie où $x\,\operatorname{ET}\,y$ vaut $1$ et
la partie où ça vaut $0$ :

```python
x = np.array([0, 1])
y = np.array([0, 1])
X, Y = np.meshgrid(x, y)
Z = np.logical_and(X, Y)

fig = plt.figure(dpi=200)

heatmap = plt.scatter(X, Y, c=Z)
plt.colorbar(heatmap)
plt.show()
```

Ici voilà les valeurs que renvoie notre neurone :

```python
x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)
X, Y = np.meshgrid(x, y)
Z = 0.5*X + 0.5*Y - 0.6 > 0

fig = plt.figure(dpi=200)

heatmap = plt.pcolormesh(X, Y, Z, shading="auto")
plt.colorbar(heatmap)
plt.show()
```

On confirme : ça marche !


Ça marche aussi très bien pour $\operatorname{OU}$ et $\operatorname{NON}$

### 🙅🏻 Exo 🙅🏻

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

L'algorithme du perceptron de Rosenblatt permet de trouver des poids pour lesquels le perceptron
simple partitionne de façon exacte un jeu de données avec un étiquetage linéairement séparable.

On considère un nombre $α>0$, le *taux d'apprentissage*.

L'algorithme prend la forme suivante :

- Initialiser le vecteur de poids $W$ à des valeurs arbitraires.
- Tant qu'il reste des points $X$ mal classifiés.

  - Pour chaque couple $(X, y) \in \mathcal{D}$ :

    - Si $y = 1$, on pose $y'=1$, et si $y=0$, $y'=-1$
    - Calculer $z = \langle W | X \rangle$.
    - Si $y'×z ≤ 0$:
      - $W←W+α×y'×X$

Notez que :

- La condition $y'×z ≤ 0$ est une façon compressée de dire “si $y'$ et $z$ sont de signes opposés” et donc
  “si $\hat{y} ≠ y$”.
- La mise à jour de $W$ réduit son écart angulaire avec $X$ si $y=1$ et l'augmente de $X$ si $y=0$.
  Pour s'neconvaincre, faire un dessin, puis calculer $\langle W+ay'X | X \rangle$.
- On peut compresser la condition et la mise à jour en une seule ligne : $W←W+α(y-\hat{y})X$.
- Pour économiser un test, on peut simplement écrire $y'=2y-1$.

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

2\. Appliquer l'algorithme du perceptron sur les données que vous venez de générer pour apprendre
les données d'entraînement (sans utiliser de terme de biais) et tester les performances du
classifieur ainsi appris sur les données de test.

3\. Représenter (par exemple avec `matplotlib.pyplot`) les données de test (en utilisant des
couleurs différentes pour les deux classes) ainsi que la frontière du classifieur.

## Perceptron multi-classe

Le cas d'un problème de classification à $n$ classes se traite en prédisant un score par classe avec
une fonction linéaire par classe et en affectant la classe pour laquelle le score est maximal.
Formellement, on dispose donc d'un $n$-uplet de poids $W_1, …, W_n$ et on a pour un exemple $X$ :

$$
\begin{align}
    z_1 &= \langle W_1 | X \rangle\\
        &⋮\\
    z_n &= \langle W_n | X \rangle
    \hat{y} &= \argmax_k z_k
\end{align}
$$

Moralement, on peut y penser comme avoir $n$ perceptrons, un par classe.

L'algorithme d'apprentissage du perceptron simple s'adapte simplement : pour les exemples mal
classifiés : on ajuste les $W_k$ de façon à ce que le score $z_y$ de la classe correcte augmente et
à ce que le score de la classe prédite diminue (puisqu'elle est incorrecte). En pseudo-code :

- Tant qu'on est pas satisfait⋅e:
  - Pour chaque $(X, y)∈\mathcal{D}$:
    - Pour chaque $k∈⟦1, n⟧$:
      - Calculer $z_k=\langle W_k | X \rangle$
    - Déterminer $\hat{y}=\argmax_k z_k$
    - Si $\hat{y}\neq y$:
      - $W_y←W_y+αX$
      - $W_{\hat{y}}←W_{\hat{y}}-αX$

Le critère de satisfaction peut être comme précédemment “jusqu'à ce qu'il n'y ait plus d'erreur”,
mais aussi comme précédemment ce n'est atteignable qu'avec des conditions contraignantes sur les
données. Un critère d'arrêt plus réaliste peut être un nombre maximal d'étapes ou l'arrêt de
l'amélioration des performances.

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
