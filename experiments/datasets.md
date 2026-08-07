v1 : bounding boxes rectangulaires; Haute-Garonne. Séparation aléatoire entre train, val, test. Problème : risque que certains terrains se retrouvent dans plusieurs datasets -> performances du modèle pas sûres.

v2 : bounding boxes rectangulaires; Haute-Garonne pour train, Gironde pour val. On va juste tester la différence entre les BB normales et orientées.

v3 : bounding boxes orientées; même split que v2

v4 : Haute-Garonne et Hautes-Pyrénées en train; 30 premiers terrains de Gironde en val.
