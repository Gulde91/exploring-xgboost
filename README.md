
## Formål
Formålet med dette projekter er empirisk at undersøge og forstå XGBoost algoritmen. 

### Data
Til at undersøge XGBoost bruges følgende datasæt:   
  - [Adult](https://rpubs.com/H_Zhu/235617)

### XGBoost og kategorisk data
Her er blevet trænet 2 xgboost modeller med binær target på features der alle er kategoriske. Forskellen på modellerne er hvordan de kategoriste features håndteret. Der er gjort på følgende måde:  
  1. De er blot taget med som numeriske features  
  2. De er one-hot encoded  
  3. Weight of evidens er benyttet  
Modellerne performer stort set ens når der måles på roc auc og accuracy. Dog er der stor forskel i træningstiden på modellerne. One-hot modellen tager omtrent 25 % længere tid at tune end modellen med de numeriske features.



### Sparse vs dense træningsdata
