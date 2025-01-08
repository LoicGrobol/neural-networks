---
title: Réseau de neurones — M2 PluriTAL 2025
layout: default
---

<!-- LTeX: language=fr -->

## News

- **2024-12-18** Les [consignes pour les projets]({{site.url}}{{site.baseurl}}/projects) sont en
  ligne.
- **2024-11-19** Premier cours du semestre le 20/11/2024

## Infos pratiques

- **Quoi** « Réseaux de neurones »
- **Où** Salle 408, bâtiment de la formation continue
- **Quand** 8 séances, les mercredi de 9:30 à 12:30, du 20/11/24 au 22/01/25
  - Voir le planning pour les dates exactes (quand il aura été mis en ligne)
- **Contact** L. Grobol [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)


**Note** ce cours remplace pour l'année 2024-2025 le cours « arbres, graphes, réseaux »

## Liens utiles

- Prendre rendez-vous pour des *office hours* en visio :
  [mon calendrier](https://calendar.app.google/N9oW2c9BzhXsWrrv9)
- Lien Binder de secours :
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LoicGrobol/neural-networks/main)
- [Consignes pour les projets]({{site.url}}{{site.baseurl}}/projects).

## Séances

Tous les supports sont sur [github](https://github.com/loicgrobol/neural-networks), voir
[Utilisation en local](#utilisation-en-local) pour les utiliser sur votre machine comme des
notebooks. À défaut, ce sont des fichiers Markdown assez standards, qui devraient se visualiser
correctement sur la plupart des plateformes (mais ne seront pas dynamiques).

Les slides et les notebooks ci-dessous ont tous des liens Binder pour une utilisation interactive
sans rien installer. Les slides ont aussi des liens vers une version HTML statique utile si Binder
est indisponible.

### 2024-11-20 — Historique et perceptron simple

- {% notebook_badges slides/01-perceptron/perceptron.py.md %}
  [Notebook perceptron simple]({{site.url}}{{site.baseurl}}/slides/01-perceptron/perceptron.py.ipynb)
  - [`perceptron_data.py`]({{site.url}}{{site.baseurl}}/slides/01-perceptron/perceptron_data.py)
  - {% notebook_badges slides/01-perceptron/perceptron-solutions.py.md %}
    [solutions]({{site.url}}{{site.baseurl}}/slides/01-perceptron/perceptron-solutions.py.ipynb)
    (partielles)


### 2024-11-27 — *Réseaux* de neurones

- {% notebook_badges slides/02-nn/nn.py.md %} [Notebook réseaux de
  neurones]({{site.url}}{{site.baseurl}}/slides/02-nn/nn.py.ipynb)
  - [`perceptron_data.py`]({{site.url}}{{site.baseurl}}/slides/01-perceptron/perceptron_data.py)


### 2024-12-04 — Représentations vectorielles des données

- [Slide CM](slides/03-representations/lecture/representations.pdf)
- {% notebook_badges slides/03-representations/lab/bias.py.md %} [Notebook TP]({{site.url}}{{site.baseurl}}/slides/03-representations/lab/bias.py.ipynb)
  - [Dossier
    `data`](https://github.com/{{site.repository}}/tree/main/slides/03-representations/lab/data/opinion-lexicon-English)
    - {% notebook_badges slides/03-representations/lab/bias-solutions.py.md %}
    [Solutions]({{site.url}}{{site.baseurl}}/slides/03-representations/lab/bias-solutions.py.ipynb)

### 2024-12-18 – Traitement de séquences

En hommage à William Labov, décédé cette nuit, un peu de lecture :


- [*The Social Motivation of a Sound Change*](https://web.stanford.edu/~eckert/PDF/LabovVineyard.pdf)
- [*Some observations on the foundations of linguistics*](https://www.ling.upenn.edu/~wlabov/Papers/Foundations.html)
- [*Empirical foundations for a theory of language change*](https://mnytud.arts.unideb.hu/tananyag/szoclingv_alap/wlh.pdf)


### 2025-01-08 —  Attention is all you need?

Article principal :

- Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, et Illia Polosukhin. 2017. « [Attention is All you Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) ». In Advances in Neural Information Processing Systems 30, édité par I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, et R. Garnett, 5998‑6008. Long Beach, California: Curran Associates, Inc.

Auxiliaires :

- Alammar, Jay. 2018. « [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) ». 2018.
- Bahdanau, Dzmitry, Kyunghyun Cho, et Yoshua Bengio. 2015. « [Neural Machine Translation by Jointly Learning to Align and Translate](http://arxiv.org/abs/1409.0473) ». In Proceedings of the 3rd International Conference on Learning Representations, édité par Yoshua Bengio et Yann LeCun. San Diego, California, USA.
- Huang, Austin, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, et Stella Biderman. 2022. « [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) ». 2022.


## Utilisation en local

Les supports de ce cours sont écrits en Markdown, convertis en notebooks avec
[Jupytext](https://github.com/mwouts/jupytext). C'est entre autres une façon d'avoir un historique
git propre, malheureusement ça signifie que pour les ouvrir en local, il faut installer les
extensions adéquates. Le plus simple est le suivant

1. Récupérez le dossier du cours, soit en téléchargeant et décompressant
   [l'archive](https://github.com/LoicGrobol/neural-networks/archive/refs/heads/main.zip)
   soit en le clonant avec git : `git clone
   https://github.com/LoicGrobol/neural-networks.git` et placez-vous dans ce dossier.
2. Créez un environnement virtuel pour le cours

   ```console
   python3 -m virtualenv .venv
   source .venv/bin/activate
   ```

3. Installez les dépendances

   ```console
   pip install -U -r requirements.txt
   ```

4. Lancez Jupyter

   ```console
   jupyter notebook
   ```

   JupyterLab est aussi utilisable, mais la fonctionnalité slide n'y fonctionne pas pour l'instant.

## Ressources

### Apprentissage artificiel

- [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/) de Daniel Jurafsky et
  James H. Martin est **indispensable**. Il est disponible gratuitement, n'hésitez pas à le
  consulter très fréquemment.
- [*Apprentissage artificiel - Concepts et
  algorithmes*](https://www.eyrolles.com/Informatique/Livre/apprentissage-artificiel-9782416001048/)
  d'Antoine Cornuéjols et Laurent Miclet. Plus ancien mais en français et une référence très
  complète sur l'apprentissage (en particulier non-neuronal). Il est un peu cher alors si vous
  voulez l'utiliser, commencez par me demander et je vous prêterai le mien.

### Python général

Il y a beaucoup, beaucoup de ressources disponibles pour apprendre Python. Ce qui suit n'est qu'une
sélection.

#### Livres

Ils commencent à dater un peu, les derniers gadgets de Python n'y seront donc pas, mais leur lecture
reste très enrichissante (les algos, ça ne vieillit jamais vraiment).

- *How to think like a computer scientist*, de Jeffrey Elkner, Allen B. Downey, and Chris Meyers.
  Vous pouvez l'acheter. Vous pouvez aussi le lire
  [ici](http://openbookproject.net/thinkcs/python/english3e/)
- *Dive into Python*, by Mark Pilgrim. [Ici](http://www.diveintopython3.net/) vous pouvez le lire ou
  télécharger le pdf.
- *Learning Python*, by Mark Lutz.
- *Beginning Python*, by Magnus Lie Hetland.
- *Python Algorithms: Mastering Basic Algorithms in the Python Language*, par Magnus Lie Hetland.
  Peut-être un peu costaud pour des débutants.
- Programmation Efficace. Les 128 Algorithmes Qu'Il Faut Avoir Compris et Codés en Python au Cours
  de sa Vie, by Christoph Dürr and Jill-Jênn Vie. Si le cours vous paraît trop facile. Le code
  Python est clair, les difficultés sont commentées. Les algos sont très costauds.

#### Web

Il vous est vivement conseillé d'utiliser un (ou plus) des sites et tutoriels ci-dessous.

- **[Real Python](https://realpython.com), des cours et des tutoriels souvent de très bonne qualité
  et pour tous niveaux.**
- [Un bon tuto NumPy](https://cs231n.github.io/python-numpy-tutorial/) qui va de A à Z.
- [Coding Game](https://www.codingame.com/home). Vous le retrouverez dans les exercices
  hebdomadaires.
- [Code Academy](https://www.codecademy.com/fr/learn/python)
- [newcoder.io](http://newcoder.io/). Des projets commentés, commencer par 'Data Visualization'
- [Google's Python Class](https://developers.google.com/edu/python/). Guido a travaillé chez eux.
  Pas [ce
  Guido](http://vignette2.wikia.nocookie.net/pixar/images/1/10/Guido.png/revision/latest?cb=20140314012724),
  [celui-là](https://en.wikipedia.org/wiki/Guido_van_Rossum#/media/File:Guido_van_Rossum_OSCON_2006.jpg)
- [Mooc Python](https://www.fun-mooc.fr/courses/inria/41001S03/session03/about#). Un mooc de
  l'INRIA, carré.
- [Code combat](https://codecombat.com/)

### Divers

- La chaîne YouTube [3blue1brown](https://www.youtube.com/c/3blue1brown) pour des vidéos de maths
  générales.
- La chaîne YouTube de [Freya Holmér](https://www.youtube.com/c/Acegikmo) plutôt orientée *game
  design*, mais avec d'excellentes vidéos de géométrie computationnelle.

## Licences

[![CC BY Licence
badge](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)

Copyright © 2023 Loïc Grobol [\<loic.grobol@gmail.com\>](mailto:loic.grobol@gmail.com)

Sauf indication contraire, les fichiers présents dans ce dépôt sont distribués selon les termes de
la licence [Creative Commons Attribution 4.0
International](https://creativecommons.org/licenses/by/4.0/). Voir [le README](README.md#Licences)
pour plus de détails.

 Un résumé simplifié de cette licence est disponible à
 <https://creativecommons.org/licenses/by/4.0/>.

 Le texte intégral de cette licence est disponible à
 <https://creativecommons.org/licenses/by/4.0/legalcode>
