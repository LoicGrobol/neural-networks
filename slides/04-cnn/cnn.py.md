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
    display_name: cours-nn
    language: python
    name: python3
---

<!-- LTeX: language=fr -->

TD : CNN pour la classification d'images
===========================================

*Inspiré du TP « [Classification d'images par
CNN](https://perso.ensta.fr/~manzaner/Cours/MI204/MI204-TPCNN.pdf) » d'[Antoine
Manzanerra](https://perso.ensta.fr/~manzaner)* et du tutoriel [Training a
Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) de Pytorch

```python
import datasets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
```

## Données

On va travailler sur [CIFAR10](https://en.wikipedia.org/wiki/CIFAR-10) qui est un classique de la
classification d'images. Commençons par le charger. Il y a plein d'options

- La source est sur [la page d'Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html).
- Il est dans
  [Torchvision](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html),
  qui est pratique pour utiliser dans Torch, mais pas le plus sympa en général.
- (Il est aussi dans [Keras](https://keras.io/api/datasets/cifar10/) mais c'est surtout utile si on vous
  oblige à utiliser Tensorflow.)
- Il est sur [Huggingface hub](https://huggingface.co/datasets/uoft-cs/cifar10), qui est pas
  parfait, mais assez pratique, on va donc utiliser ça.


Pour ça, on exécute simplement la cellule ci-dessous, attention c'est un peu long (la première fois, après c'est dans le cache).

```python
cifar10 = datasets.load_dataset("uoft-cs/cifar10")
print(cifar10)
```

La structure de donnée est un peu différente des `Bunch` de scikit-learn, essentiellement on a un dictionnaire de splits :

```python
print(cifar10["train"])
print(cifar10["test"])
```

Chacun de ces trucs est un genre de Dataframe (d'ailleurs on peut les convertir en Dataframe) :

```python
print(len(cifar10["train"]["img"]))
print(cifar10["train"][0])
```

Les images sont stockées au format PIL de la bibliothèque [Pillow](https://python-pillow.github.io/), qui est assez standard. On peut les visualiser avec [`Ipython.display.display`](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html) :

```python
img0 = cifar10["train"]["img"][0]  # La façon standard d'accéder aux données c'est par colonne
display(img0)
```

On peut aussi les récupérer comme des Array Numpy (grace à la magie de l'[interface Array](https://numpy.org/doc/stable/reference/arrays.interface.html)) :

```python
arr0 = np.array(img0)
print(arr0.shape)
print(arr0)
```

Dans quel format : pas si facile à trouver dans la doc, mais ça dépend du
[mode](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes). Ici :

```python
print(img0.mode)
```

> RGB (3x8-bit pixels, true color)

Donc on a un tableau de taille $X×Y×d$, le pixel en position $(2, 7)$ est accessible comme :

```python
print(arr0[2, 7])
```

Il est représenté par un vecteur d'entiers entre $0$ et $256$ (autrement dit des *8 bits unsigned
integers* ou `uint8`), représentant les intensités respecives des couleurs rouge, vert et bleu pour
ce pixel. Allez lire
[Wikipedia](https://en.wikipedia.org/wiki/RGB_color_model#Numeric_representations).


Et les labels ?

```python
print(cifar10["train"]["label"][0])
```

Pas très instructif, mais d'après [la doc du dataset](https://huggingface.co/datasets/uoft-cs/cifar10#data-fields) :

> label: 0-9 with the following correspondence 0 airplane 1 automobile 2 bird 3 cat 4 deer 5 dog 6 frog 7 horse 8 ship 9 truck


On va donc se doter d'un lexique pour les présenter mieux :

```python
classes_names = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
```


```python
print(classes_names[cifar10["train"]["label"][0]])
```

Et comme pour un dataframe, on peut créer une colonnes pour ça :

```python
cifar10 = cifar10.map(lambda r: {"class": classes_names[r["label"]]})
print(cifar10)
```

Des fois, si le dataset est de taille raisonnable, ça peut être intéressant de travailler
directement avec toute une colonne vue comme un itérable de valeurs. Ici par exemple pour calculer les Z-scores :

```python
# On calcule la moyenne et l"écart-type de façon brutale : avec un dataset qui est plus gros que la
# mémoire ça ne marcherait pas (mais on pourrait le faire sur un échantillon au besoin ou utiliser
# une truc comme LayerNorm)

# On récupère toutes les images sous forme d'arrays avec la conversion en série `with_format`
train_imgs = cifar10.with_format("numpy")["train"]["img"]
# On applatit tout pour juste avoir l'équivalent d'une liste de pixels et on les convertit de uint8
# en float pour pouvoir calculer avec
pixels = np.concatenate([t for t in train_imgs]).astype(np.float64).reshape((-1, 3))
# On calcule moyennes et écarts-type. Attention : on devra utiliser les mêmes pour normaliser le
# test.
train_means = np.mean(pixels, axis=0)
train_stds = np.std(pixels, axis=0, mean=train_means.reshape((1, -1)))
print(train_stds, train_means)

# Pour normaliser on va le faire avec map. ATTENTION! Conv2d utilise la disposition contre-intuitive
# (channels, height, width) donc on va aussi devoir transposer
cifar10 = cifar10.with_format("numpy").map(
    lambda row: {"zdata": np.transpose((row["img"] - train_means) / train_stds, axes=(2, 1, 0))}
)
print(cifar10)
print(cifar10["train"]["zdata"][0])
```

## Modèle

Un réseau convolutionnel de base :


```python
class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Une seule couche d'activation et un seul pooling pour tout le réseau : pourquoi ce n'est
        # pas un problème ? Parce qu'elles n'ont pas de paramètres entraînables.
        self.activation = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        # Nombre de channels dans la dernière convolution×taille de l'image après deux max-pooling
        # par bloc de taille 2
        self.fc1 = torch.nn.Linear(in_features=16 * 5 * 5, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=128)
        self.fc3 = torch.nn.Linear(in_features=128, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ATTENTION! Conv2d utilise la disposition contre-intuitive (batch, channels, height,
        # width), mais on a fait le job ci-dessus pour que les zdata soient bien orientés.
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view((x.shape[0], -1))  # flatten all dimensions except batch
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


print(ConvNet())
```

Qu'on entraîne

```python
torch.manual_seed(0)
model = ConvNet()

# On enregistre les poids initiaux pour plus tard :
torch.save(model, "weights_init.pt")

optim = torch.optim.SGD(model.parameters(), lr=0.03)

print("Epoch\tLoss")

loss_history = []
epoch_length = cifar10["train"].num_rows
for epoch in range(8):
    epoch_loss = 0.0
    # On parcourt le dataset en utilisant la conversion automatique des images vers des tenseurs
    # voir https://huggingface.co/docs/datasets/use_with_pytorch#other-feature-types
    for i, row in enumerate(cifar10["train"].with_format("torch")):
        # Unsqueeze pour faire des batchs de taille 1
        output = model(row["zdata"].to(torch.float32).unsqueeze(0))
        # Multi-classif : cross-entropy (ou log_softmax puis nll_loss)
        loss = torch.nn.functional.cross_entropy(output.view(-1), row["label"])
        loss.backward()
        optim.zero_grad()

        # Pour l'affichage toujours
        epoch_loss += loss.item()
        if i % 512 == 0:
            print(f"{epoch + i / epoch_length:.3f}\t{loss.item()}")
    loss_history.append(epoch_loss)
    print(f"{epoch + 1.0}\t{epoch_loss / epoch_length}")
```

Tracez la courbe d'appretissage :

```python

```

## Évaluation


Évaluer les performances du modèle entraîné sur quelques exemples. Est-ce satisfaisant ?

```python

```

Calculer l'exactitude globale sur le train et sur le test. Est-ce satisfaisant ?

```python

```

Calculer les matrices de confusions et l'exactitude par classe :

```python

```

## Bricolage

Testez différents hyperparamètres et notez les effets (ou absence d'effets). Essayez par exemple :

- De changer la profondeur et la largeur du réseau
- D'y ajouter des couches de normalisation comme `Dropout` ou `LayerNorm`
- De ne pas normaliser les données
- Différents taux d'apprentissages
- D'utiliser `Adam` au lieu de `SGD`
- De travailler par mini-batch (un peu de travail pour celui-là). Essayez différentes tailles.
- Faire du *early stopping* sur un ensemble de dev
- 
- …

Mettez vos tests dans des cellules indépendantes ci-dessous. Prenez des notes sur les sources que
vous utilisez et les résultats que vous trouvez. N'hésitez pas à écrire des fonctions utilitaires
(par exemple pour la boucle d'entraînement).

L'état de l'art sur CIFAR-10 a un taux d'erreur de 0.5% (avec des trucs sophistiqués). Avec des
efforts raisonnables, ça devrait être possible de faire au plus 40%, mais c'est aussi possible de
faire beaucoup mieux.

# IV - Visualisation des zones d'activation


```python
from keras.models import Model

reduced_model = Model(inputs=model.inputs, outputs=model.layers[1].output)
reduced_model.summary()
```


```python
feature_maps = reduced_model.predict(x_test)
```


```python
def get_mask(k):
    feature_maps_positive = np.maximum(feature_maps[k], 0)
    mask = np.sum(feature_maps_positive, axis=2)
    mask = mask / np.max(mask)
    return mask
```


```python
random_ids = np.random.choice(len(x_test), n_display, replace=False)
f, rd_img = plt.subplots(1, n_display, figsize=(16, 16))
for k in range(n_display):
    img = x_test_initial[random_ids[k]]
    rd_img[k].imshow(img)
    rd_img[k].axis("off")
f, rd_maps = plt.subplots(1, n_display, figsize=(16, 16))
for k in range(n_display):
    mask = get_mask(random_ids[k])
    rd_maps[k].imshow(mask)
    rd_maps[k].axis("off")
```


```python

```
