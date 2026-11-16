# Pronostic de RUL de roulements – chaîne reproductible + capteur simulé Node-RED

Données ACTUELLES = SIMULÉES (simulate.py). À remplacer par PRONOSTIA/FEMTO-ST avant la présentation (voir bas de page).

## Contenu
| Fichier | Rôle |
|---|---|
| simulate.py | Flotte de 14 roulements simulés (3 conditions), format PRONOSTIA simplifié (1 point / 10 s) |
| rul_core.py | Indicateur de santé, détection d'onset, modèle c + A·exp(kτ), bootstrap par blocs -> distribution de RUL |
| run_validation.py | Séparation par unités, réglage sur train, calibration, métriques, politiques de décision, figures |
| api_server.py | API Flask temps réel (POST /sample, /reset) utilisée par Node-RED |
| nodered_flow.json | Flow Node-RED : capteur simulé + injection de défauts + dashboard |

## Lancer
    pip install numpy scipy pandas matplotlib flask
    python run_validation.py        # reproduit tableaux + figures (graine fixe = 7)
    python api_server.py            # puis, dans Node-RED : Menu > Import > nodered_flow.json
    # Node-RED : palette -> installer node-red-dashboard ; dashboard sur http://localhost:1880/ui

## Protocole (anti-fuite)
- Séparation PAR ROULEMENT : 9 train / 5 test (conditions 2,3,1,3,1). Aucun point d'un roulement test n'est vu à l'apprentissage.
- Hyperparamètres (fenêtre, seuil d'onset, points minimum) réglés sur le train seulement ; le prior de population utilise les durées de vie du train (leave-one-out en train).
- Évaluation en ligne et CAUSALE : à l'instant t, seules les données <= t sont utilisées (médiane glissante causale).
- Calibration de l'incertitude (gonflement du bruit bootstrap) apprise sur le train pour viser 80 % de couverture.

## Passer aux vraies données PRONOSTIA (à faire !)
1. Télécharger le jeu FEMTO-ST/PRONOSTIA (IEEE PHM 2012 Data Challenge). Vérifier la structure des fichiers acc_*.csv (colonne d'accélération horizontale).
2. Pour chaque fichier acc_XXXXX.csv : RMS de la colonne horizontale -> un point du vecteur x du roulement.
3. Implémenter load_pronostic() dans simulate.py, remplacer make_fleet(), régler FT (justifier : seuil 20 g de PHM 2012 sur le signal brut, ou RMS) et refaire le découpage train/test par roulement.
4. Relancer run_validation.py : toutes les figures/tables se régénèrent.
