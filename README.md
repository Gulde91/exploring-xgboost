Readme
================

### Formål

Formålet med dette projekter er empirisk at undersøge og forstå XGBoost
algoritmen.

#### Data

Til at undersøge XGBoost bruges følgende datasæt:  
\- [Adult](https://rpubs.com/H_Zhu/235617)

#### XGBoost og kategorisk data

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

#### Sparse vs dense træningsdata
