Readme
================

## Formål

Formålet med dette projekter er empirisk at undersøge og forstå visse
grene af XGBoost algoritmen.

### Data

Til at undersøge XGBoost bruges følgende datasæt:  
\- [Adult](https://rpubs.com/H_Zhu/235617)

### XGBoost og kategorisk data

Her er blevet trænet 3 xgboost modeller med binær target på features der
alle er kategoriske på adult-datasættet. Forskellen på modellerne er
hvordan de kategoriske features er håndteret. Der er gjort på følgende
måde:  
1\. De er blot taget med som numeriske features  
2\. De er one-hot encoded  
3\. Weight of evidens er benyttet

Modellerne er blevet tunet med random search metoden, med en tune length
på 100, på følgende parametre; eta, max\_depth, gamma,
min\_child\_weight, subsample, colsample\_bytree, lambda.

Træningstiderne for de 3 modeller kan ses her:  
1\. Model med numeriske features: 302.08 sekunder  
2\. Model med one hot encoded features: 345.94 sekunder  
3\. Model med weight of evidence encoded features: 271 sekunder

En roc kurve med tilhørende auc værdier for hver af de 3 modeller ses i
plottet.  
![](./results/cat_exp_roc_plot.jpg)

Til sidst er accuracy udregnet med et cutoff på 0.5 for hver model.  
1\. Model med numeriske features: 0.8277  
2\. Model med one-hot encoded features: 0.8236  
3\. Model med weight of evidens encoded features: 0.8312

### Sparse vs dense træningsdata

Her er blevet trænet 2 modeller på identisk datagrundlag (hvor
kategoriske features er one-hot-encoded). Forskellen ligger i
dataformatet til XGB-modellen. I det ene tilfælde er input til modellen
en sparse matrix (dgCMatrix) og i det andet tilfælde er input en dense
matrix (matrix). Den primære interesse er at finde ud af, hvor stor
tidsforskellen er på at træne en XGB-model på henholdsvis sparse og
dense data. Modellerne tunes med 5 fold crossvalidation over det samme
random search grid med længden 50 på parametrene; eta, max\_depth,
gamma, min\_child\_weight, subsample, colsample\_bytree, lambda.
Træningstiden kan ses her:  
\- Model på sparse data: 180.14 sekunder  
\- Model på dense data: 540.07 sekunder  
Som der kan ses er modellen tunet på sparse data omtrent 3 gange
hurtigere end modellen tunet på dense data.

### Missing data

Xgboost kan som standard godt håndtere NA i data. Gain ved et split
bliver udregnet ud fra data som ikke er NA og de resterende samples med
NA bliver derefter assignet til den optimale direction. Der er dog den
hage ved det, at hvis input er en sparse matrice, så bliver 0’er i data
behandlet som NA. Her undersøges betydningen af at have NA i data og i
hvilket format data er i (sparse eller dense matrix). Her imputeres NA i
data og det gøres først random og bagefter systematisk. Kategoriske
features er one-hot-encoded.

##### Random imputation af NA

Her er 10%, 20%, 30% og 40% af data imputeret med NA. Her er trænet 2
modeller på samme data og med samme random search grid (der kan
naturligvis være forskel på hvilke parametre der er optimale) for hvert
niveau af NA procent-delen. Det ene datasæt er encoded som en sparse
matrix og det andet som en dense matrix.

En roc kurve med tilhørende auc værdier for hver af de 2 modeller ses i
plottet.  
<img src="./results/sparse_vs_dense_random_na_roc_plot_10_percent.jpg" width="50%" /><img src="./results/sparse_vs_dense_random_na_roc_plot_20_percent.jpg" width="50%" /><img src="./results/sparse_vs_dense_random_na_roc_plot_30_percent.jpg" width="50%" /><img src="./results/sparse_vs_dense_random_na_roc_plot_40_percent.jpg" width="50%" />

Til sidst er accuracy udregnet med et cutoff på 0.5 for hver model.

1.  Model med sparse encoding og 10% NA 0.8216
2.  Model med dense encoding og 10% NA 0.8229
3.  Model med sparse encoding og 20% NA 0.8214
4.  Model med dense encoding og 20% NA 0.8205
5.  Model med sparse encoding og 30% NA 0.8123
6.  Model med dense encoding og 30% NA 0.8122
7.  Model med sparse encoding og 40% NA 0.8085
8.  Model med dense encoding og 40% NA 0.8094

##### Systematisk imputation af NA

Her er xx % af data imputeret med NA. Her er trænet 2 modeller på samme
data og med samme search grid (der kan naturligvis være forskel på
hvilke parametre der er optimale). Det ene datasæt er encoded som en
sparse matrix og det andet som en dense matrix.

En roc kurve med tilhørende auc værdier for hver af de 2 modeller ses i
plottet.  
![](./results/cat_exp_roc_plot.jpg) <!-- indsæt rigtigt plot -->

Til sidst er accuracy udregnet med et cutoff på 0.5 for hver model.  
1\. Model med sparse data: 0.8277  
2\. Model med dense data: 0.8236
