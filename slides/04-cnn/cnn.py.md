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
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        # Une seule couche d'activation et un seul pooling pour tout le réseau : pourquoi ce n'est
        # pas un problème ? Parce qu'elles n'ont pas de paramètres entraînables.
        self.activation = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

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

# Apprendre XOR n'est pas si rapide, on va faire 50 000 epochs
loss_history = []
for epoch in range(8):
    # Pour l'affichage
    epoch_loss = 0.0
    # On parcourt le dataset en utilisant la conversion automatique des images vers des tenseurs
    # voir https://huggingface.co/docs/datasets/use_with_pytorch#other-feature-types
    for i, row in enumerate(cifar10["train"].with_format("torch")):
        # Unsqueeze pour faire des batchs de taille 1
        output = model(row["zdata"].to(torch.float32).unsqueeze(0))
        # Multi-classif : cross-entropy (ou log_softmax puis nll_loss)
        loss = torch.nn.functional.cross_entropy(output.view(-1), row["label"])
        loss.backward()
        optim.step()
        # On doit remettre les gradients des paramètres à zéro, sinon ils
        # s'accumulent quand on appelle `backward`
        optim.zero_grad()
        # Pour l'affichage toujours
        epoch_loss += loss.item()
        if i % 512 == 0:
            print(f"{epoch + i / cifar10['train'].num_rows:.3f}\t{loss.item()}")
    loss_history.append(epoch_loss)
    print(f"{epoch + 1.0}\t{epoch_loss / cifar10['train'].num_rows}")

```

## Évaluation


Évaluer les performances du modèle entraîné sur quelques exemples. Est-ce satisfaisant ?

```python
random_ids = np.random.choice(len(x_test), n_display, replace=False)
pred = np.argmax(model.predict(x_test[random_ids]), axis=1)
f, axarr = plt.subplots(1, n_display, figsize=(16, 16))
for k in range(n_display):
    axarr[k].imshow(x_test_initial[random_ids[k]])
    axarr[k].title.set_text(classes[pred[k]])
```

Vos premiers résultats semblent-ils corrects ?

Hum ! Affichons à présent la précision sur l'ensemble de votre base :


```python
print(
    "Précision du réseau sur les {} images d'entraînement : {:.2f} %".format(
        n_training_samples, 100 * history_dict["acc"][-1]
    )
)
print(
    "Précision du réseau sur les {} images de validation : {:.2f} %".format(
        n_valid, 100 * history_dict["val_acc"][-1]
    )
)
```


```python
def accuracy_per_class(model):
    n_classes = len(classes)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    pred = np.argmax(model.predict(x_test), axis=1)
    for i in range(len(y_test)):
        confusion_matrix[np.argmax(y_test[i]), pred[i]] += 1

    print("{:<10} {:^10}".format("Classe", "Précision (%)"))
    total_correct = 0
    for i in range(n_classes):
        class_total = confusion_matrix[i, :].sum()
        class_correct = confusion_matrix[i, i]
        total_correct += class_correct
        percentage_correct = 100.0 * float(class_correct) / class_total
        print("{:<10} {:^10.2f}".format(classes[i], percentage_correct))
    test_acc = 100.0 * float(total_correct) / len(y_test)
    print("Précision du réseau sur les {} images de test : {:.2f} %".format(len(y_test), test_acc))
    return confusion_matrix


confusion_matrix = accuracy_per_class(model)
```

### III.2. Matrices de Confusion

Les matrices de confusion nous renseignent plus précisément sur la nature des erreurs commises par notre modèle.


```python
# Plot normalized confusion matrix
plot_confusion_matrix(
    confusion_matrix, classes, normalize=True, title="Matrice de confusion normalisée"
)

# Plot non-normalized confusion matrix
plot_confusion_matrix(confusion_matrix, classes, title="Matrice de confusion non normalisée")
```

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
