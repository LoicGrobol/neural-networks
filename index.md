---
title: Réseau de neurones — M2 PluriTAL 2026
layout: default
---

<!-- LTeX: language=fr -->

## News

- **2025-11-25** Premier cours du semestre le 26/11/2025

## Infos pratiques

- **Quoi** « Réseaux de neurones »
- **Où** Salle R06, bâtiment de la formation continue
- **Quand** 8 séances, les mercredi de 9:30 à 12:30, du 20/11/24 au 22/01/25
  - Voir le planning pour les dates exactes (quand il aura été mis en ligne)
- **Contact** L. Grobol [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)


## Liens utiles

- Prendre rendez-vous pour des *office hours* en visio :
  [mon calendrier](https://calendar.app.google/N9oW2c9BzhXsWrrv9)

## Séances

Les liens dans chaque séance vous permettent de télécharger les fichiers `.ipynb` à utiliser (et
données additionnelles éventuelles). Attention : pour les utiliser en local, il faudra installer les
packages du `requirements.txt` (dans un environnement virtuel). Si vous ne savez pas comment faire,
allez voir [« Utilisation en local »](#utilisation-en-local)

### 2025-11-26 — Historique et perceptron simple

- {% notebook_badges slides/01-perceptron/perceptron.py.md %}
  [Notebook perceptron simple]({{site.url}}{{site.baseurl}}/slides/01-perceptron/perceptron.py.ipynb)
  - [`perceptron_data.py`]({{site.url}}{{site.baseurl}}/slides/01-perceptron/perceptron_data.py)
  - {% notebook_badges slides/01-perceptron/perceptron-solutions.py.md %}
    [solutions]({{site.url}}{{site.baseurl}}/slides/01-perceptron/perceptron-solutions.py.ipynb)
    (partielles)

## Utilisation en local

### Environnements virtuels et packages

Je cite le [Crash course Python](slides/01-tools/python_crash_course.py.ipynb):

- Les environnements virtuels sont des installations isolées de Python. Ils vous permettent d'avoir
  des versions indépendantes de Python et des packages que vous installez
  - Gérez vos environnements et vos packages avec [uv](https://docs.astral.sh/uv/). Installez-le,
 lisez la doc.
  - Pour créer un environnement virtuel : `uv venv /chemin/vers/…`
  - La convention, c'est `uv venv .venv`, ce qui créée un dossier (caché par défaut sous Linux et
 Mac OS car son nom commence par `.`) : `.venv` dans le dossier courant (habituellement le
 dossier principal de votre projet). Donc faites ça.
  - Il est **obligatoire** de travailler dans un environnement virtuel. L'idéal est d'en avoir un
 par cours, un par projet, etc. - uv est assez précautionneux avec l'espace disque, il y a donc
 assez peu de désavantage à avoir beaucoup d'environnements virtuels.
  - Un environnement virtuel doit être **activé** avant de s'en servir. Concrètement ça remplace la
 commande `python` de votre système par celle de l'environnement. - Dans Bash par exemple, ça se
 fait avec `source .venv/bin/activate` (en remplaçant par le chemin de l'environnement s'il est
   différent) - `deactivate` pour le désactiver et rétablir votre commande `python`. À faire
 avant d'en activer un autre.
- On installe des packages avec `uv pip` ou `python -m pip` (mais plutôt `uv pip`, et jamais juste
  `pip`).
  - `uv pip install numpy` pour installer Numpy.
  - Si vous avez un fichier avec un nom de package par ligne (par exemple le
 [`requirements.txt`](https://github.com/LoicGrobol/web-interfaces/blob/main/requirements.txt) du
 cours) : `uv pip install -U -r requirements.txt`
  - Le flag `-U` ou `--upgrade` sert à mettre à jour les packages si possible : `uv pip install -U
	numpy` etc.
- Je répète : on installe uniquement dans un environnement virtuel, on garde ses environnements bien
  séparés (un par cours, pas un pour tout le M2).
  - Dans un projet, on note dans un `requirements.txt` (ou `.lst`) les packages dont le projet a
 besoin pour fonctionner.
  - Les environnements doivent être légers : ça ne doit pas être un problème de les effacer, de les
 recréer… Si vous ne savez pas recréer un environnement que vous auriez perdu, c'est qu'il y a un
 problème dans votre façon de les gérer.
- Si vous voulez en savoir plus, **et je recommande très fortement de vouloir en savoir plus, c'est
  vital de connaître ses outils de travail**, il faut : *lire les documentations de **tous** les
  outils et **toutes** les commandes que vous utilisez*.

Maintenant à vous de jouer :

- Installez uv
- Créez un dossier pour ce cours
- Dans ce dossier, créez un environnement virtuel nommé `.venv`
- Activez-le
- Téléchargez le
  [`requirements.txt`](https://github.com/LoicGrobol/web-interfaces/blob/main/requirements.txt)
  et installez les packages qu'il liste

### Notebooks Jupyter

Si vous avez une installation propre (par exemple en suivant les étapes précédentes), vous pouvez
facilement ouvrir les notebooks du cours :

- Téléchargez le notebook du [Crash course
  Python](https://loicgrobol.github.io/apprentissage-artificiel/slides/01-tools/python_crash_course.py.ipynb)
  et mettez-le dans le dossier que vous utilisez pour ce cours.
- Dans un terminal (avec votre environnement virtuel activé) lancez jupyter avec `jupyter notebook
  python_crash_course.py.ipynb`.
- Votre navigateur devrait s'ouvrir directement sur le notebook. Si ça ne marche pas, le terminal
  vous donne dans tous les cas un lien à suivre.

Alternativement, des IDE comme vscode permettent d'ouvrir directement les fichiers ipynb. Pensez à
lui préciser que le kernel à utiliser est celui de votre environnement virtuel s'il ne le trouve pas
tout seul.

### Utilisation avancée

Vous pouvez aussi (mais je ne le recommande pas forcément car ce sera plus compliqué pour vous de le
maintenir à jour) cloner [le dépôt du
cours](https://github.com/loicgrobol/apprentissage-artificiel). Tous les supports y sont, sous forme
de fichiers Markdown assez standards, qui devraient se visualiser correctement sur la plupart des
plateformes. Pour les utiliser comme des notebooks, il vous faudra utiliser l'extension
[Jupytext](https://github.com/mwouts/jupytext) (qui est dans le `requirements.txt`). C'est entre
autres une façon d'avoir un historique git propre.

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

Copyright © 2025 L. Grobol [\<lgrobol@parisnanterre.fr\>](mailto:lgrobol@parisnanterre.fr)

Sauf indication contraire, les fichiers présents dans ce dépôt sont distribués selon les termes de
la licence [Creative Commons Attribution 4.0
International](https://creativecommons.org/licenses/by/4.0/). Voir [le README](README.md#Licences)
pour plus de détails.

 Un résumé simplifié de cette licence est disponible à
 <https://creativecommons.org/licenses/by/4.0/>.

 Le texte intégral de cette licence est disponible à
 <https://creativecommons.org/licenses/by/4.0/legalcode>
