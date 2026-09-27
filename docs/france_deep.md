# France deep dive: France vs US+India (test)

## A) Volume, density, prediction rate at t=0.75

country | n_s1 | mean_rec_per_s1 | matches_per_s1 | pct_zero_pred
---|---|---|---|---
India | 809986 | 35.865 | 3.218 | 6.262
France | 259452 | 33.415 | 3.370 | 5.326
US | 663106 | 34.959 | 3.329 | 5.831

Share of records with max p in [0.3, 0.9]:

country | pct_mid
---|---
France | 4.53
India | 5.12
US | 2.30

US+India matches/S1 = 3.2684. France t matching that rate: **t=0.940** (France matches/S1 at t=0.75 = 3.3695).

## B) Legal-form tokens: first/last, caught vs not

**France S1 (raw) — top 40 first tokens** (token, count, caught_by_legal_regex):

maison(7376,N), ets(5253,N), établissements(5169,N), bordeaux(5169,N), centre(4700,N), école(4593,N), nantes(4465,N), lille(4204,N), association(4004,N), amicale(3907,N), fédération(3877,N), comité(3875,N), club(3809,N), union(3807,N), clinique(2415,N), lycée(2411,N), pharmacie(2382,N), institut(2372,N), collège(2370,N), ehpad(2364,N), la(2258,N), tourcoing(2174,N), dunkerque(2106,N), roubaix(2053,N), calais(1955,N), saint-nazaire(1659,N), pessac(1468,N), deleves(1390,N), mérignac(1292,N), team(1098,N), chasse(1004,N), lège-cap-ferret(989,N), pornic(977,N), communale(887,N), saint-herblain(859,N), cercle(716,N), publique(627,N), nationale(602,N), sapeurs(566,N), defense(555,N)

**France S1 (raw) — top 40 last tokens** (token, count, caught_by_legal_regex):

sarl(73443,Y), sas(52252,Y), eurl(16967,Y), sa(12745,Y), sasu(10712,Y), sci(8357,Y), ei(4182,Y), club(2617,N), jean(1713,N), (france)(1576,N), ecole(1480,N), amicale(1444,N), comite(1395,N), sainte(1304,N), sportive(1154,N), amis(961,N), pierre(957,N), cie(876,N), notre(848,N), centre(811,N), dame(764,N), frères(763,N), fils(762,N), parents(713,N), marie(703,N), union(645,N), paul(631,N), primaire(610,N), societe(570,N), fetes(561,N), martin(516,N), anciens(514,N), compagnie(509,N), pharmacie(508,N), joseph(499,N), loisirs(473,N), maison(471,N), louis(456,N), sante(447,N), deleves(411,N)

**France records (raw, S2+S3) — top 40 first tokens** (token, count, caught_by_legal_regex):

maison(32377,N), sarl(27013,Y), ets(23638,N), bordeaux(23203,N), établissements(23011,N), nantes(20142,N), club(20053,N), amicale(19441,N), centre(19234,N), lille(19019,N), association(18444,N), union(18413,N), école(18332,N), sas(18212,Y), comité(18122,N), fédération(18025,N), la(10576,N), pharmacie(10003,N), lycée(9852,N), tourcoing(9782,N), clinique(9725,N), collège(9717,N), institut(9546,N), dunkerque(9538,N), roubaix(9284,N), ehpad(9115,N), calais(8747,N), saint-nazaire(7517,N), eurl(6990,Y), pessac(6698,N), deleves(6349,N), mérignac(5805,N), sa(5785,Y), team(5055,N), sasu(5009,Y), chasse(4493,N), lège-cap-ferret(4473,N), pornic(4450,N), sci(4367,Y), communale(4061,N)

**France records (raw, S2+S3) — top 40 last tokens** (token, count, caught_by_legal_regex):

sarl(217993,Y), sas(153358,Y), eurl(65586,Y), sa(54670,Y), sasu(49869,Y), sci(44180,Y), groupe(24716,N), fils(23319,N), france(22283,N), développement(22026,N), club(22004,N), participations(13445,N), (france)(13392,N), international(13376,N), cie(13363,N), holding(13338,N), distribution(13309,N), ecole(12905,N), amicale(12614,N), s.a.s(11773,N), comite(11741,N), s.a.s.(11658,N), snc(11456,Y), sàrl(11284,N), s.a.r.l.(11004,N), ei(10640,Y), services(10277,N), associés(9769,N), sportive(9598,N), e.u.r.l.(8504,N), amis(8236,N), centre(7332,N), s.a.(6765,N), parents(6371,N), union(6282,N), s.a.s.u.(5850,N), primaire(5485,N), s.c.i.(4951,N), societe(4871,N), fetes(4732,N)

% France S1 names ending in an uncaught frequent last-token: **11.05%**

## C) "(France)" marker

Share with marker: France S1 = 8.014%, France records = 6.477%

has_marker | median_p | n
---|---|---
False | 0.9991 | 1314600
True | 0.9992 | 115756

## D) 5-digit tokens (postal codes)

% France S1 addresses with a 5-digit token: 0.39%; % France record addresses: 0.50%

Among argmax pairs with p>=0.9 (n=3295): record contains S1's 5-digit token in **71.14%**
Among argmax pairs with p in [0.3,0.9) (n=447): record contains S1's 5-digit token in **68.01%**

## E) Abbreviation-like token pairs on high-confidence France pairs (p>=0.98)

n pairs at p>=0.98: 825949

**name** top 50 (rec_token, s1_token, count):

(cb,club,270), (maisondesant,maison,264), (center,centre,152), (cub,club,89), (farmacie,pharmacie,88), (fs,fils,83), (groupe,groupement,79), (as,amis,76), (centremdical,centre,69), (centrehospitalier,centre,69), (clb,club,68), (maisonsant,maison,62), (centremdicalde,centre,61), (centrehospitalierde,centre,60), (clbu,club,49), (cu,club,47), (nantesclub,nantes,46), (services,service,44), (culb,club,40), (cbu,club,39), (centrehospitaliersainte,centre,38), (centremdicalsainte,centre,36), (bordeauxclub,bordeaux,35), (centremdicaldu,centre,33), (tablissements,etablissements,33), (ais,amis,32), (un,union,31), (pharmaciedela,pharmacie,31), (cl,club,31), (nantesecole,nantes,30), (centremdicalsaint,centre,30), (ehpaddela,ehpad,30), (centrehospitaliersaint,centre,29), (lyci,lycee,29), (matemelle,maternelle,29), (ams,amis,28), (ai,amis,28), (ecle,ecole,28), (ec,ets,27), (na,nantes,26), (ep,ets,25), (lnstitut,institut,25), (teatre,theatre,25), (jn,jean,24), (bordeauxclubsarl,bordeaux,24), (fls,fils,24), (etablissement,etablissements,24), (cliniquedela,clinique,24), (institutsainte,institut,24), (comit,comite,23)

**address** top 50 (rec_token, s1_token, count):

(crs,cours,2359), (q,quai,2124), (b,bis,943), (003,3,421), (res,residence,416), (03,3,399), (005,5,367), (01,1,364), (04,4,361), (02,2,355), (001,1,353), (06,6,347), (006,6,336), (004,4,327), (0011,11,321), (007,7,318), (05,5,316), (08,8,311), (002,2,307), (07,7,306), (008,8,304), (0010,10,301), (aveue,ave,297), (0012,12,296), (016,16,293), (alee,allee,293), (09,9,293), (012,12,277), (pass,passage,272), (014,14,272), (avnue,ave,257), (aveneu,ave,256), (0013,13,252), (010,10,251), (aveune,ave,250), (011,11,247), (009,9,238), (0014,14,237), (0016,16,234), (017,17,232), (019,19,224), (avene,ave,223), (0015,15,220), (018,18,219), (013,13,217), (015,15,211), (t,ter,210), (0017,17,207), (0021,21,205), (0019,19,204)

## F) Accent folding examples (raw -> normalised)

- S1 raw=`<< Team Ecole` -> norm=`team ecole`
- S1 raw=`Maison de Santé Generation` -> norm=`maison de sante generation`
- S1 raw=`Saint-Herblain Societe SARL` -> norm=`saint herblain societe`
- S1 raw=`Securite Darts Sport SARL` -> norm=`securite darts sport`
- S1 raw=`École primaire Sainte Pierre` -> norm=`ecole primaire sainte pierre`
- S1 raw=`Rotary Sport SASU` -> norm=`rotary sport`
- S1 raw=`Bouliste (France) Amicale SARL` -> norm=`bouliste france amicale`
- S1 raw=`Clinique Saint Francois` -> norm=`clinique saint francois`
- S1 raw=`Vegan Groupe EI` -> norm=`vegan groupe`
- S1 raw=`Établissements Demployeurs SARL` -> norm=`etablissements demployeurs`
- rec raw=`SCI Ptit Àmicale` -> norm=`ptit amicale`
- rec raw=`sci ligue ici parents` -> norm=`ligue ici parents`
- rec raw=`OZT ÀMICALE SAS` -> norm=`ozt amicale`
- rec raw=`Association du Pàrenthese` -> norm=`association du parenthese`
- rec raw=`Production Thê Jeunes` -> norm=`production the jeunes`
- rec raw=`Institut du Groupe Raid` -> norm=`institut du groupe raid`
- rec raw=`Unite (France) Sante SASU` -> norm=`unite france sante`
- rec raw=`Établissements Dëleves EURL` -> norm=`etablissements deleves`
- rec raw=`Clinique Sâinte` -> norm=`clinique sainte`
- rec raw=`Appel & Frèrhs SAS` -> norm=`appel frerhs`

## G) 30 random France pairs with p in [0.3, 0.9)

- p=0.367 | S1 `Comites Federation SAS` | `35 Rue Descartes, Bordeaux, Nouvelle-Aquitaine`  vs rec `comites jeunes sas` | `40 R Descartes, Bordeaux, Nouvelle-Aquitaine`
- p=0.330 | S1 `WAA Compagnie SAS` | `20 Bis RUE de la Carterie, Nantes, Pays de la Loire`  vs rec `WAA CLUB` | `20 BIS RUE DE LA CATRERIE, Pays de la Loire, NANTES`
- p=0.759 | S1 `Ad Anciens SARL` | `10 Impasse Jean Renoir, Saint-Herblain, Pays de la Loire`  vs rec `Ad Anciens SARL` | ``
- p=0.379 | S1 `Deleves Centre SAS` | `54 Avenue des Ondines, La Baule-Escoublac, Pays de la Loire`  vs rec `Deleves SAS & Associés` | `54 Av. Des Ondines, La Baule-escoublac`
- p=0.391 | S1 `Dutilisation Leonard Élémentaire SARL` | `166 RUE Camille Godard, Bordeaux, Nouvelle-Aquitaine`  vs rec `Dutilisation Leonard Élémentaire France SARL` | `168 RUE CAMILLE GODARD, BORDEAUX`
- p=0.349 | S1 `Dunkerque Anciens SAS` | `15 RUE du Comptoir Linier, Dunkerque, Hauts-de-France`  vs rec `Dunkerque Collectif SAS` | `15 RUE DU COMPTOIR LINIER, DUNKERQUE`
- p=0.716 | S1 `Defense (France) Jeunes SARL` | `118 BIS Rue Royale, Lille, Hauts-de-France`  vs rec `Defense (France) Federation SARL` | `118 RUE FAIDHERBE, LILLE, Hauts-de-France`
- p=0.687 | S1 `ZB Jeunes SARL` | `134 BD Albert Brandenburg, Bordeaux, Nouvelle-Aquitaine`  vs rec `ZB Jeunes SARL` | ``
- p=0.303 | S1 `BPC Union SARL` | `Nantes, 36 Rue Jean Jacques Audubon, Pays de la Loire`  vs rec `BCC Union` | `36 R. Jean Jacques Audubon, Nantes, Loire-Atlantique`
- p=0.825 | S1 `Calais Culture SARL` | `Calais, 220 Rue Anatole France, Hauts-de-France`  vs rec `CALAIS INSTITUT` | `#220 RUE ANATOLE FRANCE, CALAIS, Hauts-de-France`
- p=0.386 | S1 `Lycée School` | `174 Rue du Perray, Nantes, Pays de la Loire`  vs rec `Ecole Sitl` | `174 Rue Du Perray, Nantes, Loire-Atlantique`
- p=0.470 | S1 `Bordeaux Lycée SASU` | `95 RUE Achard, Bordeaux, Nouvelle-Aquitaine`  vs rec `Bordeaux-Lycée` | `55 Rue Achard, Bordeaux`
- p=0.474 | S1 `Livre Federation SARL` | `9 Rue de l'Amiral Prouhet, Pessac, Nouvelle-Aquitaine`  vs rec `Livre  Sportive` | `09 R De L'amiral Prouhet, Pessac, Gironde`
- p=0.768 | S1 `Volvestre Saad Club SCI` | `12 Rue Olympe de Gouges, Saint-Herblain, Pays de la Loire`  vs rec `Volvestre Saad SCI & Associés` | `12 Rue Olympe De Gouges, St.-herblain, Loire-Atlantique`
- p=0.466 | S1 `Com (France) Sante SARL` | `7 Rue Alfred de Musset, Calais, Hauts-de-France`  vs rec `Com (France)  SARL & Associés` | `7 RUE ALFRED DE MUSSET, CALAIS, Pas-de-Calais`
- p=0.411 | S1 `Nature Sportive SARL` | `5 Impasse du Grand Parc, Saint-Nazaire, Pays de la Loire`  vs rec `Nature Sportive Sàrl` | ``
- p=0.339 | S1 `3eme Societe` | `17 PL des reignaux, hotel UP, Lille, Hauts-de-France`  vs rec `3EME LOISIRS` | `LILLE, HOTEL UP, 19 PL DES REIGNAUX`
- p=0.556 | S1 `Lisle Federation SARL` | `37 Boulevard Jean Ingres, Nantes, Pays de la Loire`  vs rec `Lisle Lycée SARL` | `38 BOULEVARD JEAN INGRES, NANTES, Loire-Atlantique`
- p=0.558 | S1 `Nationale Amis (France)` | `39 Rue des Soupirants, Calais, Hauts-de-France`  vs rec `NATIONALE AMIS (FRANCE) FRANCE` | `Nº 44 RUE DES SOUPIRANS, CALAIS, Pas-de-Calais`
- p=0.457 | S1 `Associations Primaire SARL` | `1 Rue Michel Manoll, Nantes, Pays de la Loire`  vs rec `Associations Primaire S.A.R.L.` | ``
- p=0.599 | S1 `Ramponneau Maison SARL` | `1 Rue Violette, Lille, Hauts-de-France`  vs rec `Ramponneau Conseil SARL` | `10 RUE VIOLETTE, Nord, LILLE`
- p=0.632 | S1 `Billard Parents (France)` | `Hauts-de-France, Tourcoing, 9 Rue du Marechal Ney`  vs rec `Billard Parents (France) SAS` | ``
- p=0.562 | S1 `EHPAD Belles` | `4 Avenue du Vallon, Mérignac, Nouvelle-Aquitaine`  vs rec `EHPAD Belles Services` | `4 Ave Du Vallon, Mérignac, Nouvelle-Aquitaine`
- p=0.470 | S1 `Roubaix Culture SASU` | `Rond point de lEurope, IUT C departement TC Universite Lille 2, Roubaix, Hauts-de-France`  vs rec `Roubaix Culture SCI` | `38 Rond Point De Leurope, Iut C Departement Tc Universite Lille 2, Roubaix, Nord`
- p=0.610 | S1 `Sophrologie Foyer EURL` | `42 Avenue de la Foret, La Teste-de-Buch, Nouvelle-Aquitaine`  vs rec `Sophrologie Foyer E.U.R.L.` | ``
- p=0.528 | S1 `Nouvelles & Fils SAS` | `83 Résidence Plein Ocean, Pornic, Pays de la Loire`  vs rec `jhclubsas.com` | `83 Residence Plein Ocean, Pornic, Pays de la Loire`
- p=0.732 | S1 `On Primaire SARL` | `20 Rue du Général Hoche, Dunkerque, Hauts-de-France`  vs rec `ON PRIMAIRE SARL` | ``
- p=0.325 | S1 `Atouts Ecole SAS` | `12 Rue des Poilus, Calais, Hauts-de-France`  vs rec `SAS Atouts Club` | `Calais, 23 Rue Des Poils, Hauts-de-France`
- p=0.727 | S1 `Tourcoing Collège` | `27 Rue Marengo, Tourcoing, Hauts-de-France`  vs rec `Tourcoing Collège SARL` | `74 Rue Marengo, Tourcoing, Nord`
- p=0.748 | S1 `Prives Classe Union SAS` | `20 Rue Cuvier, Roubaix, Hauts-de-France`  vs rec `PRIVES CLASSE LYCEE SAS` | `N° 20 Rue Cuvier, Roubaix, Hauts-de-France`
