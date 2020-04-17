# Deep Learning - Analyse de sentiments multi-langues

Présentation Equipe : 
* Florian Caliz, 
* Aubin Porte, 
* Paul Loublier, 
* Juliette Verlaine.

__Contexte__ : Développer et mettre en place un modèle permettant de détecter des sentiments sur du texte, d'abord en anglais et si possible, sur une langue autre.


## Instanciation du projet

* Etape 1 : Télécharger le dataset des données sources [ici](http://help.sentiment140.com/for-students)
* Etape 2 : Lancer le notebook _transform.ipynb_ pour effectuer le nettoyage et la transformation des données. Ce script générera un fichier proper_df.csv.
* Etape 3 a. : Soit vous décidez d'entraîner le modèle, auquel cas, lancer le fichier _main.py_
* Etape 3 b. : Soit vous décidez de lancer directement le modèle sur les données de test téléchargées précédemment pour voir comment il se comporte face à de nouvelles données. Pour cela, utiliser le fichier _load_model.py_

## Introduction

Ce premier projet de Deep Learning est un projet visant à mettre en place un modèle capable de classifier du texte et de prédire des labels. Sur un jeu de données pré annoté, le modèle doit s'entraîner
à classifier le texte à savoir s'il est négatif ou positif, et une fois entraîné avec une bonne accuracy, il doit être capable sur du texte brute, sans annotation, de prédire la valeur du label associé. 

## Transformation des données

### a. Dataset

Le dataset choisi est un dataset qui regroupe des tweet divers et variés, pré anotés. Le dataset initial est sur la base d'un .tsv. Les différentes colonnes sont : 

* 0 - La polarité du tweet (0 = negative, 2 = neutral, 4 = positive).
* 1 - L'id du tweet (2087).
* 2 - La date du tweet (Sat May 16 23:58:44 UTC 2009).
* 3 - La query (lyx). S'il n'y en a pas, la cellule est remplie par : NO_QUERY.
* 4 - L'utilisateur qui a rédigé le tweet (robotickilldozr).
* 5 - Le contenu du tweet sous la forme de texte (Lyx is cool).

### b. Transformation

Dans un premier temps, il s'agit de s'assurer que nous disposons des données nécessaires pour l'application d'un modèle de classification. Pour alléger les traitements, il faut aussi s'assurer que nous ne disposons pas de données qui n'ont pas d'intérêt dans la mise en place de notre modèle.
On constate assez rapidement que les colonnes id, date, query et user n'ont aucun impact quand à la détection de sentiment que fera notre modèle. De ce fait, nous les supprimons.

Maintenant, il s'agit de déterminer les caractères ou string qui pourraient poluer l'application du modèle : url, caractères spéciaux, citation des utilisateurs (@robotickilldozr) ou encore les #.
On garde la valeur des #thisissocool mais on enlève le symbole # qui pourrait poluer l'analyse qui suivra. Pour ce faire on utilise des regex : 

```
df['text'] = df['text'].str.replace('http\\S+|www.\\S+', '', case=False)
df['text'] = df['text'].str.replace('@\\S+', '', case=False)
```

__Etape de visualisation__ : L'étape de visualisation permet de comprendre la répartition des classes, ici positif et négatif (pas de classe neutre) mais aussi l'occurence des mots selon les labels. 
Les nuages de mots, par labels, permettent de comprendre si des textes annotés comme positifs ont une forte occurence mot positif. Ou si des mots à forte occurence sont présents mais totalement neutre et du coup, sans impact sur l'apprentissage du modèle. 

__Application de fonctions de nettoyages__ : 

```
def cleanHtml(sentence)
def cleanPunc(sentence)
def keepAlpha(sentence)
```

On nettoie ici toute trace de HTML, tout signe de ponctuation et enfin on applique la mise en minuscule de tous les caractères afin d'applanir le texte. 

__Lemmatization et Tokenization__ :

Le processus de « lemmatisation » consiste à représenter les mots (ou « lemmes » 😉) sous leur forme canonique. Par exemple pour un verbe, ce sera son infinitif. Pour un nom, son masculin singulier. L'idée étant encore une fois de ne conserver que le sens des mots utilisés dans le corpus.
La tokenisation est l'acte de décomposer une séquence de chaînes en morceaux tels que des mots, des mots-clés, des phrases, des symboles et d'autres éléments appelés jetons. Les jetons peuvent être des mots, des phrases ou même des phrases entières. Dans le processus de tokenisation, certains caractères comme les signes de ponctuation sont supprimés. Les jetons deviennent l'entrée d'un autre processus comme l'analyse et l'exploration de texte.

```
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
    proper_df['quote_lemmatizer'] = proper_df.quote.apply(lemmatize_text)
```

## Modèle

## Critique du modèle

## Conclusion

L'analyse de sentiment, lorsqu'elle comporte deux classes, ici negative et positive n'est pas un cas d'usage qui nécessite obligatoirement d'utiliser des modèles de Deep Learning. 
En effet, l'entraînement est coûteux et les résultats escomptés ne sont pas toujours à la hauteur des attentes. 

Dans le cadre de ce projet, ayant rencontrés quelques problèmes, nous avons pu tester un modèle de Machine Learning : Naive Bayes qui remontait quasiment les mêmes résultats pour un temps d'entraînement bien moins long
et coûteux en terme de performance machine. Il est donc important de bien comprendre les domaines d'application dans lequel il fait sens d'utiliser du deep learning et dans lequel il fait sens d'utiliser du machine learning. Tout dépend de la nature des données en entrée, et ce n'est pas par ce que c'est du Deep Learning que c'est pour autant magique :) ! 