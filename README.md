# Assignement ECOVADIS

## Structure

This repo is composed of five major directories :  
 >   - core which contains the utilitary functions, the Transformer, Pipeline, MLP, MetModels classes 
 >   - Notebooks with all the descriptive notebooks  
 >   - data with the excels data  
 >   - models with the saved trained model  
 >   - UI with the streamlit app for the user interface
 >   - endpoints with the flask app for deployment 


##  Baselines 

### Analysis 
The notebook analysis.ipynb contains the exploratory analysis of the data :  
>    - basic dataframe description, statistics (average median ...), missing values  
 >   - Plot distributions   
 >   - Correlation analysis   

### Baselines models
The notebooks RFBaselines.ipynb / XGBoostBaselines.ipynb / Catboost.ipynb contains :  
 >   - Data preprocessing ( categorization and normalisation ...)  
 >   - The training and bayesian hyperparmaters tunning of Random Forest, Catboost and Xgboost models
 >   - SHAP for interpreting models

 Best balanced accuracy obtained with XGBoost around 69% with 5 fold cross validation


## Beyond Simple classification 

### Analysis
The notebook Feedback.ipynb contains the explanatory of textual feedback analyis 
>   - Wordcloud
>   - LDA analysis
>   - Dictionary based approach

Xgboos retrained with labels coming from the dictionary based approach ==> no significant improvement 

Best balanced accuracy obtained with XGBoost around 73% balanced accuracy with 5 fold cross validation


### Bert Fine tuning for sentiment analysis

Google colab notebook provided in Notebooks/TrainBert
>   - Unsupervised labelisation using dictionary based approach
>   - Finetuning of Roberta model pretrained on IMDB reviews on sentiment analysis
>   - Retraining of Xboost with labels for reviews 


Current best score :  XGBoost + feature engineering + bert sentiment analysis (79.19% balanced accuracy with 5fold cross validation )
---------------

## Unsuccessful tried approaches

>    - SMOTE for oversampling negative class
>    - Clustering of review using a Doc2vec approach / Umap projection
>    - MLP on tabular dataset with over / under sampling + class weightening
>    - Meta Model using mix Catboost and XGBoost



## Deployment 
>    - Flask api with endpoint to request trained model (bert + xgboost), endpoint is available at 127.0.0.1:5000/predict
>    - Streamlit based user interface for inferance and interpretation, UI is available at localhost:8501

### To run the user interface

##### Install required librairies 
#####  Donwload model at https://drive.google.com/file/d/1qdQNi17UOp4P0qq_qb7u40C97yLWHjQL/view?usp=drive_link and put it in the models/ folder
##### Run :
```
make run-flask
make run-UI

```
### To stop the user interface

```
make clear_app
```

<br>
<br>

![DemoStreamlit](streamlit.gif)