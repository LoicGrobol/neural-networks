---
title: Projets réseaux de neurons — M2 PluriTAL 2025
layout: default
permalink: /projects/
---

[comment]: <> "LTeX: language=fr"

Projets
=======

Votre travail sera de réaliser une application, une ressource, une interface ou une bibliothèque
pour Python. Son thème devra être lié au TAL ou au traitement de données, utiliser un système de
réseaux de neurones artificiels et pourra évidemment être en lien avec d'autres cours, d'autres
projets ou votre travail en entreprise (mais assurez-vous avant que ce soit OK de partager votre
code avec moi dans ce cas).

*Bien entendu, rien ne vous empêche de combiner ces options.*

Je m'attends plus à ce que vous réalisiez un projet autonome (c'est en général plus simple), mais
votre travail peut aussi prendre la forme d'un plugin/add-on/module… pour un projet existant (pensez
à spaCy par exemple) voire une contribution substantielle à projet existant (si vous faites passer
un gros *pull request* à Pytorch par exemple) mais si c'est ce que vous visez, dites le moi très en
avance et on en discute.

Dans tous les cas, je m'attends à un travail substantiel (pas juste suivre un tuto, quoi), si vous
avez un doute, là encore, *demandez-moi*.

Pas de prompt engineering, pas de chatbots, vous en mangez assez ailleurs.

## Consignes

- Composition des groupes et sujets des projets à envoyer avant le 20 janvier 2025 (envoyer un mail
  par groupe avec vos noms, prénoms et établissements et une description concise, mais précise du
  projet). Si vous avez un doute sur la pertinence ou la faisabilité du projet, venez m'en parler
  avant.
- Projet à rendre le 8 mars 2025 *au plus tard*
- Projet de préférence collectif, par groupe de 2 ou 3
  - Si c'est un problème pour vous, venez me voir, tout est négociable
  - S'il y a un problème — quel qu'il soit — dans votre groupe, n'hésitez pas à m'en parler
- Rendus par mail à `lgrobol@parisnanterre.fr` avec en objet `[NN2025] Projet final` et les noms,
  prénoms et établissements de tous les membres du groupe dans le corps du mail.
  - **Si l'objet est différent, je ne verrai pas votre rendu**. Et si un nom manque, vous risquez de
    ne pas avoir de note.
  - J'accuserai réception sous trois jours ouvrés dans la mesure du possible, relancez-moi si ce
    n'est pas le cas.


Le rendu devra comporter :

1. Une documentation du projet traitant les points suivants :

   - Les objectifs du projet
   - Les données utilisées (origine, format, statut juridique) et les traitements opérés sur
     celles-ci
   - La méthodologie (comment vous vous êtes répartis le travail, comment vous avez identifié les
     problèmes et les avez résolus, différentes étapes du projet…)
   - L'implémentation ou les implémentations (modélisation le cas échéant, modules et/ou API
     utilisés, différents langages le cas échéant)
   - Les résultats (fichiers output, visualisations…) et une discussion sur ces résultats (ce que
     vous auriez aimé faire et ce que vous avez pu faire par exemple)

   On attend de la documentation technique, pas une dissertation. Elle pourra prendre le format d'un
   ou plusieurs fichiers, d'un site web, d'un notebook de démonstration, à votre convenance

   **La documentation ne doit pas, jamais, sous aucun prétexte, comporter de capture d'écran de
   code.**

2. Le code Python et les codes annexes (JS par ex.) que vous avez produit. Le code *doit* être
   commenté. Des tests, ce serait bien. **Évitez les notebooks**, préférez les interfaces en ligne
   de commande ou web (ou graphiques si vous êtes très motivé⋅es)

3. Les éventuelles données en input et en output (ou un échantillon si le volume est important)

N'hésitez pas à vous servir de git pour versionner vos projets !

## Conseils

Écrivez ! Tenez un carnet : vos questions, un compte rendu de vos discussions, les problèmes
rencontrés, tout est bon à prendre et cela vous aidera à rédiger la documentation finale.

## Ressources

### Données géo-localisées

Il existe beaucoup de choses pour travailler avec des données géo-localisées. Allez voir en vrac :
[Geo-JSON](http://geojson.org/), [uMap](http://umap.openstreetmap.fr/fr/) pour créer facilement des
cartes en utilisant les fonds de carte d'OpenStreetMap, [leaflet](http://leafletjs.com/) une lib JS
pour les cartes interactives, [overpass turbo](http://overpass-turbo.eu/) pour interroger facilement
les données d'OpenStreetMap (il y a une [api !](http://www.overpass-api.de/)).

### Ressources linguistiques

N'hésitez pas à aller fouiller dans [Ortolang](https://www.ortolang.fr/) ou
[Clarin](https://lindat.mff.cuni.cz/repository/xmlui/) des ressources linguistiques exploitables
librement et facilement. Vous pouvez aussi aller voir du côté de l'API twitter pour récupérer des
données (qui ne sont pas nécessairement uniquement linguistiques)

### Open Data

Quelques sources : [Paris Open Data](https://opendata.paris.fr),
[data.gouv.fr](https://data.gouv.fr), [Google dataset
search](https://toolbox.google.com/datasetsearch)

## Exemples de sujets

À gros grain :

- Appliquer des modèles existants à des nouvelles données.
- Comparer différents modèles existants sur des données existantes ou proposer de nouveaux modèles
  qui font mieux.
- Analyser, expliquer, sonder des modèles existants pour mettre à jour des explications sur leur
  manière de fonctionner ou révéler leurs failles. Vous pouvez aller voir ce qui se fait dans le
  workshop [BlackboxNLP](https://blackboxnlp.github.io/) pour vous donner des idées. (par exemple
  j'ai beaucoup aimé [*Universal Adversarial Triggers for Attacking and Analyzing
  NLP*](https://www.aclweb.org/anthology/D19-1221/)).

- Classification de documents : le classique, se décline en analyse de sentiment ; détection de
  discours haineux, de *fake news*.
- Syntaxe : POS tagging, parsing, chunking… Attention, [la
  concurrence](https://github.com/npdependency/npdependency) est rude.
- Génération de texte : résumé automatique, traduction, ASR (mais attention tout ça c'est long)
- …

Commencez plutôt petit : voyez si vous pouvez reproduire l'état de l'art en piquant du code qui
existe, puis modifiez-le pour arriver à vos fins.

On le répète encore une fois : si vous avez des idées, mais que vous galérez, si vous n'avez pas
d'idée, s'il y a un problème **contactez moi**.