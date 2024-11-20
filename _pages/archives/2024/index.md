---
title: Réseau de neurones — M2 PluriTAL 2024
layout: default
permalink: /2024/
---

<!-- LTeX: language=fr -->

## News

- **2023-11-21** Premier cours du semestre le 22/11/2023
- **2023-11-19** Les [consignes pour les projets]({{site.url}}{{site.baseurl}}/projects) sont en
  ligne.

## Infos pratiques

- **Quoi** « Réseaux de neurones »
- **Où** Salle 410, bâtiment de la formation continue
- **Quand** 8 séances, les mercredi de 9:30 à 12:30, du 23/11/22 au 25/01/23
  - Voir le planning pour les dates exactes (quand il aura été mis en ligne)
- **Contact** Loïc Grobol [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)


**Note** ce cours remplace pour l'année 2023-2024 le cours « arbres, graphes, réseaux »

## Liens utiles

- Prendre rendez-vous pour des *office hours* en visio :
  [Calendly](https://calendly.com/lgrobol/remote-office-hour)
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

### 2023-11-22 — Historique et perceptron simple

- {% notebook_badges slides/01-perceptron/perceptron.py.md %}
  [Notebook perceptron simple]({{site.url}}{{site.baseurl}}/slides/01-perceptron/perceptron.py.ipynb)
  - [`perceptron_data.py`]({{site.url}}{{site.baseurl}}/slides/01-perceptron/perceptron_data.py)

### 2023-12-05 — Réseaux de neurones

- {% notebook_badges slides/02-nn/nn.py.md %} [Notebook réseaux de
  neurones]({{site.url}}{{site.baseurl}}/slides/02-nn/nn.py.ipynb)
  - [`perceptron_data.py`]({{site.url}}{{site.baseurl}}/slides/01-perceptron/perceptron_data.py)

### 2023-12-12 — Représentations vectorielles de mots

- {% notebook_badges slides/03-embeddings/embeddings.py.md %} [Notebook réseaux de
  neurones]({{site.url}}{{site.baseurl}}/slides/03-embeddings/embeddings.py.ipynb)

### 2023-12-20 — Représentation de séquences et mécanisme d'attention

### 2024-01-10 — Transformers

Illustrations :

<!-- LTeX: language=en-GB -->

- Alammar, Jay. 2018. ‘The Illustrated Transformer’. 2018.
  <http://jalammar.github.io/illustrated-transformer/>.
- Huang, Austin, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, and Stella Biderman. 2022. ‘The
  Annotated Transformer’. 2022. <http://nlp.seas.harvard.edu/annotated-transformer/>.

<!-- LTeX: language=fr -->


Articles fondateurs :

<!-- LTeX: language=en-GB -->

- Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. 2016. ‘Layer Normalization’. In .
  <http://arxiv.org/abs/1607.06450>.
- Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. 2015. ‘Neural Machine Translation by Jointly
  Learning to Align and Translate’. In Proceedings of the 3rd International Conference on Learning
  Representations, edited by Yoshua Bengio and Yann LeCun. San Diego, California, USA.
  <http://arxiv.org/abs/1409.0473>.
- Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
  Kaiser, and Illia Polosukhin. 2017. ‘Attention Is All You Need’. In Advances in Neural Information
  Processing Systems 30, edited by I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S.
  Vishwanathan, and R. Garnett, 5998–6008. Long Beach, California: Curran Associates, Inc.
  <https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>.

<!-- LTeX: language=fr -->

Compléments :
<!-- LTeX: language=en-GB -->

- Collobert, Ronan, and Jason Weston. 2008. ‘A Unified Architecture for Natural Language Processing:
  Deep Neural Networks with Multitask Learning’. In Proceedings of the 25th International Conference
  on Machine Learning, 160–67. ICML ’08. Association for Computing Machinery.
  <https://doi.org/10.1145/1390156.1390177>.
- Collobert, Ronan, Jason Weston, Léon Bottou, Michael Karlen, Koray Kavukcuoglu, and Pavel P.
  Kuksa. 2011. ‘Natural Language Processing (Almost) from Scratch’. Journal of Machine Learning
  Research 12 (August): 2493−2537.
- Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
  2019. ‘BERT: Pre-Training of Deep - Bidirectional Transformers for Language Understanding’. In
  Proceedings of the 2019 Conference of the North American Chapter of the Association for
  Computational Linguistics: Human Language Technologies, 4171–86. Association for Computational
  Linguistics. <https://doi.org/10.18653/v1/N19-1423>.
- Radford, Alec, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. ‘Improving Language
  Understanding by Generative Pre-Training’. Technical report. OpenAI.
  <https://openai.com/blog/language-unsupervised/>.

<!-- LTeX: language=fr -->


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
