<html>
  <body>
    <h1 align="center">Classification des scenes naturelles dans les images<br>(Intel Image classification Challenge)<br> processus initie a partir de zero et valide avec des modele pre-entraines</h1>
    <img align="center" src="images/dataset.png" />
    <hr>
	<h2>Reconnaissance et detection d'objet - Travaux pratiques </h2>
    <h4>explication a propos du Reseau de Neurone(CNN)</h4>
    <p>
      & emsp; Dans les réseaux de neurones, le réseau de neurones convolutifs (ConvNets ou CNN) est l'une des principales catégories de reconnaissance d'images, de classification d'images. Les détections d'objets, les visages de reconnaissance, etc., sont quelques-uns des domaines dans lesquels les CNN sont largement utilisés. <br>
      & emsp; Le nom «<i> réseau neuronal convolutif </i>» indique que le réseau utilise une opération mathématique appelée convolution. Les réseaux convolutifs sont un type spécialisé de réseaux de neurones qui utilisent la convolution à la place de la multiplication matricielle générale dans au moins une de leurs couches. <br>
      & emsp; Les classifications d'images CNN prennent une image d'entrée, la traitent et la classent dans certaines catégories. Les ordinateurs voient une image d'entrée comme un tableau de pixels et cela dépend de la résolution de l'image. En fonction de la résolution de l'image, il verra h x l x d (h = hauteur, w = largeur, d = dimension). Par exemple, une image d'un tableau 6 x 6 x 3 de matrice RVB (3 se réfère aux valeurs RVB) et une image d'un tableau 4 x 4 x 1 de matrice d'image en niveaux de gris. <br>
      & emsp; Les CNN sont des versions régularisées de perceptrons multicouches. Les perceptrons multicouches signifient généralement des réseaux entièrement connectés, c'est-à-dire que chaque neurone d'une couche est connecté à tous les neurones de la couche suivante. La «connectivité totale» de ces réseaux les rend sujets à un surapprentissage des données. Les moyens typiques de régularisation incluent l'ajout d'une forme de mesure de la grandeur des poids à la fonction de perte.
    </p>
    <h3><a href="https://en.wikipedia.org/wiki/Convolutional_neural_network">1. CNN Architecture</a></h3>
    <h4>1. a) Convolutional</h4>
    <h4>1. b) Pooling</h4>
    <h4>1. c) Fully connected</h4>
    <h4>1. d) Receptive field</h4>
    <h4>1. e) Weights</h4>
    <hr>
    <h2>L'apprentissage par transfert à partir de modèles pré-formés</h2>
    <p> & emsp; L'intuition derrière l'apprentissage par transfert pour la classification d'images est que si un modèle est entraîné sur un ensemble de données suffisamment grand et général, ce modèle servira effectivement de modèle générique du monde visuel. Nous pouvons ensuite tirer parti de ces cartes d'entités apprises sans avoir à recommencer à zéro en entraînant un grand modèle sur un grand ensemble de données. <br> <br> & emsp; L'apprentissage par transfert est une optimisation, un raccourci pour gagner du temps ou obtenir de meilleures performances. En général, il n'est pas évident qu'il y aura un avantage à utiliser l'apprentissage par transfert dans le domaine tant que le modèle n'aura pas été développé et évalué. Lisa Torrey et Jude Shavlik dans <a href="http://amzn.to/2fgeVro"> leur chapitre sur l'apprentissage par transfert </a> décrivent trois avantages possibles à rechercher lors de l'utilisation de l'apprentissage par transfert: <ul>
      <li> <strong> Début plus élevé </strong> - La compétence initiale (avant d'affiner le modèle) sur le modèle source est plus élevée qu'elle ne le serait autrement
      <li> <strong> Pente plus élevée </strong>: le taux d'amélioration des compétences lors de la formation du modèle source est plus élevé qu'il ne le serait autrement
      <li> <strong> Asymptote plus élevée </strong>: la compétence convergée du modèle entraîné est meilleure qu'elle ne le serait autrement
    </ul> </p>
    <h3>1. ResNet50</h3>
    <p>
       & emsp; ResNet (Residual Networks) -50 est une variante du modèle ResNet qui comporte 48 couches de convolution avec 1 MaxPool et 1 Average Pool. Nous pouvons charger une version pré-entraînée du réseau formée sur plus d'un million d'images de la base de données ImageNet. Un modèle ResNet50 a été pré-entraîné sur un million d'images de la base de données ImageNet et peut classer les images en 1000 catégories d'objets. En conséquence, le réseau a appris de riches représentations de caractéristiques pour une large gamme d'images. Le réseau a une taille d'entrée d'image de 224 x 224. La percée fondamentale avec ResNet a été qu'il nous a permis d'entraîner avec succès des réseaux de neurones extrêmement profonds avec plus de 150 couches. Avant la formation ResNet, les réseaux de neurones très profonds étaient difficiles en raison du problème de la disparition des gradients.
     </p>
    <h3>2. VGG16</h3>
    <p>
       & emsp; VGG (Visual Geometric Group) 16 est un modèle de réseau de neurones convolutif proposé par K. Simonyan et A. Zisserman de l'Université d'Oxford dans l'article «<i> Very Deep Convolutional Networks for Large-Scale Image Recognition </i> ». Le modèle atteint une précision de test de 92,7% dans le top 5 dans ImageNet, qui est un ensemble de données de plus de 14 millions d'images appartenant à 1000 classes. C'était l'un des modèles célèbres soumis à <a href="http://www.image-net.org/challenges/LSVRC/2014/results"> ILSVRC-2014 </a>. Le 16 dans VGG16 fait référence à 16 couches qui ont des poids. Ces 16 couches contiennent les paramètres entraînables et il existe d'autres couches comme la couche de pool Max, mais celles-ci ne contiennent aucun paramètre entraînable. Ce réseau est un assez grand réseau et il a environ 138 millions de paramètres (environ).
     </p>
    <h3>3. InceptionV3</h3>
    <p>
      & emsp; Inception-v3 est une architecture de réseau neuronal convolutif de la famille Inception qui apporte plusieurs améliorations, notamment l'utilisation du lissage d'étiquettes, des convolutions factorisées 7 x 7 et l'utilisation d'un classificateur auxiliaire pour propager les informations d'étiquette plus bas sur le réseau (avec l'utilisation de la normalisation des lots pour les couches de la tête latérale). En repensant l'architecture de départ, on obtient une efficacité de calcul et moins de paramètres. Avec moins de paramètres, un réseau d'apprentissage en profondeur de 42 couches, avec une complexité similaire à VGGNet, peut être atteint. Avec 42 couches, un taux d'erreur plus faible est obtenu et en fait le premier finaliste de la classification des images dans <a href="http://www.image-net.org/challenges/LSVRC/"> ILSVRC </a> ( ImageNet Large Scale Visual Recognition Competition) 2015. Inception-V3 a été formé à l'aide d'un ensemble de données de 1 000 classes de l'ensemble de données ImageNet original qui a été formé avec plus d'un million d'images d'entraînement, la version Tensorflow a 1 001 classes, ce qui est dû à un «arrière-plan» supplémentaire classe non utilisée dans ImageNet d'origine.
    </p>
  </body>
</html>
