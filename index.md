---
title: Réseau de neurones — M2 PluriTAL 2022
layout: default
---

<!-- LTeX: language=fr -->

## News

- **2023-01-16** Les [consignes pour les projets]({{site.url}}{{site.baseurl}}/projects) sont en
  ligne.
- **2022-11-21** Premier cours du semestre le 23/11/2022

## Infos pratiques

- **Quoi** « Réseaux de neurones »
- **Où** Salle 219, bâtiment Paul Ricœur
- **Quand** 8 séances, les mercredi de 9:30 à 12:30, du 23/11/22 au 25/01/23
  - Voir le planning pour les dates exactes (quand il aura été mis en ligne)
- **Contact** Loïc Grobol [<lgrobol@parisnanterre.fr>](mailto:lgrobol@parisnanterre.fr)


**Note** ce cours remplace pour l'année 2022-2023 le cours « arbres, graphes, réseaux »

## Liens utiles

- Lien Binder de secours :
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LoicGrobol/neural-networks/main)
- [Consignes pour les projets]({{site.url}}{{site.baseurl}}/project) sont en ligne.

## Séances

Tous les supports sont sur [github](https://github.com/loicgrobol/neural-networks), voir
[Utilisation en local](#utilisation-en-local) pour les utiliser sur votre machine comme des
notebooks. À défaut, ce sont des fichiers Markdown assez standards, qui devraient se visualiser
correctement sur la plupart des plateformes (mais ne seront pas dynamiques).

Les slides et les notebooks ci-dessous ont tous des liens Binder pour une utilisation interactive
sans rien installer. Les slides ont aussi des liens vers une version HTML statique utile si Binder
est indisponible.

### 2022-11-23 — Historique et perceptron simple

- {% notebook_badges slides/01-perceptron/perceptron-slides.py.md %}
  [Notebook perceptron simple](slides/01-perceptron/perceptron-slides.py.ipynb)

### 2022-12-07 - Perceptron multi-couches

- {% notebook_badges slides/02-nn/nn-slides.py.md %}
  [Notebook perceptron multi-couches](slides/02-nn/nn-slides.py.ipynb)

### 2023-01-03 — Transformers

- {% notebook_badges slides/03-transformers/transformers-slides.py.md %}
  [Notebook demo](slides/03-transformers/transformers-slides.py.ipynb)
- [Cours 🤗 Transformers](https://huggingface.co/course)

### 2023-01-10 — Descente de gradient et *backpropagation*

Un peu de lecture supplémentaire :

- Pour les maths et les visuels, la leçon [*backpropagation
  calculus*](https://www.3blue1brown.com/lessons/backpropagation-calculus) de 3blue1brown est bien.
- Pour l'historique, l'article [*Backpropagation*](https://en.wikipedia.org/wiki/Backpropagation) de
  Wikipedia en est pas mal.
- Pour une zoologie des variantes de l'algo de descente de gradient, [Sebastian
  Ruder](https://ruder.io/optimizing-gradient-descent/) a fait une bonne synthèse

### 2023-01-18 — Modèles de génération de séquences

Articles cités :

<!-- LTeX: language=en-US -->

- Adelani, David, Jesujoba Alabi, Angela Fan, Julia Kreutzer, Xiaoyu Shen, Machel Reid, Dana Ruiter,
  et al. 2022. “[A Few Thousand Translations Go a Long Way! Leveraging Pre-Trained Models for
  African News Translation.](https://doi.org/10.18653/v1/2022.naacl-main.223)” In Proceedings of the
  2022 Conference of the North American Chapter of the Association for Computational Linguistics:
  Human Language Technologies, 3053–70. Seattle, United States: Association for Computational
  Linguistics.
- Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. 2015. “[Neural Machine Translation by Jointly
  Learning to Align and Translate.](http://arxiv.org/abs/1409.0473)” In Proceedings of the 3rd
  International Conference on Learning Representations, edited by Yoshua Bengio and Yann LeCun. San
  Diego, California, USA.
- Fan, Angela, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep
  Baines, et al. 2021. “Beyond English-Centric Multilingual Machine Translation.” The Journal of
  Machine Learning Research 22 (1): 107:4839-107:4886.
- Levy, Omer, Kenton Lee, Nicholas FitzGerald, and Luke Zettlemoyer. 2018. “[Long Short-Term Memory
  as a Dynamically Computed Element-Wise Weighted Sum.](http://arxiv.org/abs/1805.03716)”
  ArXiv:1805.03716 [Cs, Stat], May.
- Lewis, Mike, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy,
  Veselin Stoyanov, and Luke Zettlemoyer. 2020. “[BART: Denoising Sequence-to-Sequence Pre-Training
  for Natural Language Generation, Translation, and
  Comprehension.](https://doi.org/10.18653/v1/2020.acl-main.703)” In Proceedings of the 58th Annual
  Meeting of the Association for Computational Linguistics, 7871–80. Online: Association for
  Computational Linguistics.
- Merrill, William, Gail Weiss, Yoav Goldberg, Roy Schwartz, Noah A. Smith, and Eran Yahav. 2020.
  “[A Formal Hierarchy of RNN Architectures.](https://www.aclweb.org/anthology/2020.acl-main.43)” In
  Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 443–59.
  Online: Association for Computational Linguistics.
- Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
  Kaiser, and Illia Polosukhin. 2017. “[Attention Is All You
  Need.](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)”
  In Advances in Neural Information Processing Systems 30, edited by I. Guyon, U. V. Luxburg, S.
  Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, 5998–6008. Long Beach, California:
  Curran Associates, Inc..
- Weiss, Gail, Yoav Goldberg, and Eran Yahav. 2018. “[On the Practical Computational Power of Finite
  Precision RNNs for Language Recognition.](https://www.aclweb.org/anthology/P18-2117)”, 2:740–45.
  Melbourne, Australia: Association for Computational Linguistics.

<!-- LTeX: language=fr -->

### 2023-01-25 — Biais, attaques, données

Articles cités :

<!-- LTeX: language=en-US -->

- Abid, Abubakar, Maheen Farooqi, and James Zou. 2021a. “Large Language Models Associate Muslims
  with Violence.” Nature Machine Intelligence 3 (6): 461–63.
  <https://doi.org/10.1038/s42256-021-00359-2>.
- Abid, Abubakar, Maheen Farooqi, and James Zou. 2021b. “Persistent Anti-Muslim Bias in Large
  Language Models.” In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society,
  298–306. AIES ’21. New York, NY, USA: Association for Computing Machinery.
  <https://doi.org/10.1145/3461702.3462624>.
- Adelani, David, Jesujoba Alabi, Angela Fan, Julia Kreutzer, Xiaoyu Shen, Machel Reid, Dana Ruiter,
  et al. 2022. “A Few Thousand Translations Go a Long Way! Leveraging Pre-Trained Models for African
  News Translation.” In Proceedings of the 2022 Conference of the North American Chapter of the
  Association for Computational Linguistics: Human Language Technologies, 3053–70. Seattle, United
  States: Association for Computational Linguistics.
  <https://doi.org/10.18653/v1/2022.naacl-main.223>.
- Artetxe, Mikel, and Holger Schwenk. 2019. “Massively Multilingual Sentence Embeddings for
  Zero-Shot Cross-Lingual Transfer and Beyond.” Transactions of the Association for Computational
  Linguistics 7 (September): 597–610. <https://doi.org/10.1162/tacl_a_00288>.
- Bender, Emily M., Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021. “On the
  Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜.” In Proceedings of the 2021 ACM
  Conference on Fairness, Accountability, and Transparency, 610–23. FAccT ’21. New York, NY, USA:
  Association for Computing Machinery. <https://doi.org/10.1145/3442188.3445922>.
- Birhane, Abeba. 2022. “ChatGPT, Galactica, and the Progress Trap.” Wired, December 9, 2022.
  <https://www.wired.com/story/large-language-models-critique/>.
- Caliskan, Aylin, Joanna J. Bryson, and Arvind Narayanan. 2017. “Semantics Derived Automatically
  from Language Corpora Contain Human-like Biases.” Science 356 (6334): 183–86.
  <https://doi.org/10.1126/science.aal4230>.
- Cheng, Pengyu, Weituo Hao, Siyang Yuan, Shijing Si, and Lawrence Carin. 2022. “FairFil:
  Contrastive Neural Debiasing Method for Pretrained Text Encoders.” In .
  <https://openreview.net/forum?id=N6JECD-PI5w>.
- Fan, Angela, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep
  Baines, et al. 2021. “Beyond English-Centric Multilingual Machine Translation.” The Journal of
  Machine Learning Research 22 (1): 107:4839-107:4886.
- Gonen, Hila, and Yoav Goldberg. 2019. “Lipstick on a Pig: Debiasing Methods Cover up Systematic
  Gender Biases in Word Embeddings But Do Not Remove Them.” In Proceedings of the 2019 Workshop on
  Widening NLP, 60–63. Association for Computational Linguistics.
  <https://www.aclweb.org/anthology/W19-3621/>.
- Gonen, Hila, Yova Kementchedjhieva, and Yoav Goldberg. 2019. “How Does Grammatical Gender Affect
  Noun Representations in Gender-Marking Languages?” In Proceedings of the 23rd Conference on
  Computational Natural Language Learning (CoNLL), 463–71. Association for Computational
  Linguistics. <https://doi.org/10.18653/v1/K19-1043>.
- Gonen, Hila, Shauli Ravfogel, and Yoav Goldberg. 2022. “Analyzing Gender Representation in
  Multilingual Models.” arXiv. <https://doi.org/10.48550/arXiv.2204.09168>.
- Gonen, Hila, and Kellie Webster. 2020. “Automatically Identifying Gender Issues in Machine
  Translation Using Perturbations.” In Findings of the Association for Computational Linguistics:
  EMNLP 2020, 1991–95. Online: Association for Computational Linguistics.
  <https://doi.org/10.18653/v1/2020.findings-emnlp.180>.
- Kreutzer, Julia, Isaac Caswell, Lisa Wang, Ahsan Wahab, Daan van Esch, Nasanbayar Ulzii-Orshikh,
  Allahsera Tapo, et al. 2022. “Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets.”
  Transactions of the Association for Computational Linguistics 10 (January): 50.
  <https://doi.org/10.1162/tacl_a_00447>.
- Liang, Sheng, Philipp Dufter, and Hinrich Schütze. 2020. “Monolingual and Multilingual Reduction
  of Gender Bias in Contextualized Representations.” In Proceedings of the 28th International
  Conference on Computational Linguistics, 5082–93. Barcelona, Spain (Online): International
  Committee on Computational Linguistics. <https://doi.org/10.18653/v1/2020.coling-main.446>.
- Lucy, Li, and David Bamman. 2021. “Gender and Representation Bias in GPT-3 Generated Stories.” In
  Proceedings of the Third Workshop on Narrative Understanding, 48–55. Virtual: Association for
  Computational Linguistics. <https://doi.org/10.18653/v1/2021.nuse-1.5>.
- Mikolov, Tomas, Wen-tau Yih, and Geoffrey Zweig. 2013. “Linguistic Regularities in Continuous
  Space Word Representations.” In Proceedings of the 2013 Conference of the North American Chapter
  of the Association for Computational Linguistics: Human Language Technologies, 746–51. Atlanta,
  Georgia: Association for Computational Linguistics. <https://aclanthology.org/N13-1090>.
- Perez, Fábio, and Ian Ribeiro. 2022. “Ignore Previous Prompt: Attack Techniques For Language
  Models.” arXiv. <https://doi.org/10.48550/arXiv.2211.09527>.
- Perrigo, Billy. 2023. “The $2 Per Hour Workers Who Made ChatGPT Safer.” Time, January 18, 2023.
  <https://time.com/6247678/openai-chatgpt-kenya-workers/>.
- Prabhakaran, Vinodkumar, Margaret Mitchell, Timnit Gebru, and Iason Gabriel. 2022. “A Human
  Rights-Based Approach to Responsible AI.” arXiv. <https://doi.org/10.48550/arXiv.2210.02667>.
- Prost, Flavien, Nithum Thain, and Tolga Bolukbasi. 2019. “Debiasing Embeddings for Reduced Gender
  Bias in Text Classification.” In Proceedings of the First Workshop on Gender Bias in Natural
  Language Processing, 69–75. Florence, Italy: Association for Computational Linguistics.
  <https://doi.org/10.18653/v1/W19-3810>.
- Ravfogel, Shauli, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. 2020. “Null It Out:
  Guarding Protected Attributes by Iterative Nullspace Projection.” In Proceedings of the 58th
  Annual Meeting of the Association for Computational Linguistics, 7237–56. Online: Association for
  Computational Linguistics. <https://doi.org/10.18653/v1/2020.acl-main.647>.
- Steed, Ryan, Swetasudha Panda, Ari Kobren, and Michael Wick. 2022. “Upstream Mitigation Is  NOt
  All You Need: Testing the Bias Transfer Hypothesis in Pre-Trained Language Models.” In Proceedings
  of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
  Papers), 3524–42. Dublin, Ireland: Association for Computational Linguistics.
  <https://doi.org/10.18653/v1/2022.acl-long.247>.
- Wallace, Eric, Shi Feng, Nikhil Kandpal, Matt Gardner, and Sameer Singh. 2019. “Universal
  Adversarial Triggers for Attacking and Analyzing NLP.” In Proceedings of the 2019 Conference on
  Empirical Methods in Natural Language Processing and the 9th International Joint Conference on
  Natural Language Processing, 2153–62. Association for Computational Linguistics.
  <https://doi.org/10.18653/v1/D19-1221>.
- Weidinger, Laura, John Mellor, Maribeth Rauh, Conor Griffin, Jonathan Uesato, Po-Sen Huang, Myra
  Cheng, et al. 2021. “Ethical and Social Risks of Harm from Language Models.” arXiv.
  <https://doi.org/10.48550/arXiv.2112.04359>.


<!-- LTeX: language=fr -->


## Lire les slides en local

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

Copyright © 2022 Loïc Grobol [\<loic.grobol@gmail.com\>](mailto:loic.grobol@gmail.com)

Sauf indication contraire, les fichiers présents dans ce dépôt sont distribués selon les termes de
la licence [Creative Commons Attribution 4.0
International](https://creativecommons.org/licenses/by/4.0/). Voir [le README](README.md#Licences)
pour plus de détails.

 Un résumé simplifié de cette licence est disponible à
 <https://creativecommons.org/licenses/by/4.0/>.

 Le texte intégral de cette licence est disponible à
 <https://creativecommons.org/licenses/by/4.0/legalcode>
