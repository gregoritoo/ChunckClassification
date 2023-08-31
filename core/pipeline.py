import numpy as np
import pandas as pd 
import pickle 
import xgboost as xgb
from .transformer import SentimentModel
from .utils import convert_df_to_numeric_df
import nltk
import shap 
from transformers_interpret import SequenceClassificationExplainer
import joblib

class Pipeline():
    def __init__(self,trained_scaler,trained_prediction_model,trained_bert_model,device="cpu"):
        self.trained_scaler = trained_scaler
        self.trained_prediction_model = trained_prediction_model
        self.trained_bert_model = trained_bert_model
        self.device = device
        self._load_bert_model()
        self._load_scaler()
        self._load_xgb_model()
        


    def _convert_csv(self,path_to_data):
        self.data = pd.read_csv(path_to_data)


    def _load_bert_model(self):
        self.bert_model = SentimentModel(self.device)
        self.bert_model.load_pretrained(self.trained_bert_model)


    def _load_scaler(self):
        with open(self.trained_scaler,"rb") as f :
            std_scaler = pickle.load(f) 
        self.std_scaler = std_scaler

    def _load_xgb_model(self):
        self.xgb_model = joblib.load(self.trained_prediction_model)
        #self.xgb_model = xgb.XGBClassifier()
        #self.xgb_model.load_model(self.trained_prediction_model)


    def scale_data(self,columns=["EstimatedSalary","Balance (EUR)","CreditScore","Age","Tenure","RatioSalary",
                                    "RatioProducst","RatioCards","ProductSalary"]):
        keys_to_keep = [column_name for column_name in self.data.keys() if column_name not in columns ]
        print(keys_to_keep)
        sub_df = self.data[columns].copy()
        df_scaled = self.std_scaler.fit_transform(sub_df.to_numpy())
        df_scaled = pd.DataFrame(df_scaled, columns=columns)
        for key in keys_to_keep :
            df_scaled[key] = self.data[key]
        self.data = df_scaled
        



    def predict_preprocessed(self,x):
        y_predicted = self.xgb_model.predict(x)
        y_predicted = np.where(y_predicted > 0.5,1,0)
        return y_predicted


    def _get_feedbacks(self):
        true_sentence = [ sentence for sentence in self.data["CustomerFeedback"].values if isinstance(sentence,str)]
        true_sentence_index = [ i for i,sentence in enumerate(self.data["CustomerFeedback"].values) if isinstance(sentence,str)]
        for position,index in enumerate(true_sentence_index) :
            assert self.data["CustomerFeedback"].iloc[index] == true_sentence[position]
        corpus_by_feedback = [ nltk.sent_tokenize(text) for text in true_sentence]
        return true_sentence, true_sentence_index, corpus_by_feedback


    def _process_feedbacks(self):
        predictions_sentiments = []
        true_sentence, true_sentence_index, corpus_by_feedback = self._get_feedbacks()
        for i,feedback in enumerate(corpus_by_feedback) :
            results = []
            for sentence in feedback :
                result = self.bert_model.get_scores(sentence).detach().cpu().numpy()
                results.append(result)
            predictions_sentiments.append(np.argmax(np.vstack(results).mean(axis=0)))
        self.data['SentimentsDictionnaryBert'] = [-1 for _ in range(len(self.data))]
        for position,index in enumerate(true_sentence_index):
            self.data.at[index,'SentimentsDictionnaryBert'] = predictions_sentiments[position]
        self.data.drop(columns=["CustomerFeedback"])


    def predict(self,data):
        self.feedbacks = data.CustomerFeedback.values
        if isinstance(data,str):
            self._convert_csv(data)
        else :
            self.data = data
        self.data = self.data.reset_index(drop=True)
        self._process_feedbacks()
        self._features_engineering()
        try :
            self.data = self.data.drop(columns=["CustomerFeedback_cat"])
        except Exception as e :
            pass
        self.data = self.data[['EstimatedSalary','Balance (EUR)','CreditScore','Age','Tenure','RatioSalary','RatioProducst',
'RatioCards','ProductSalary','NumberOfProducts','HasCreditCard','IsActiveMember','SentimentsDictionnaryBert','Country_cat','Gender_cat']]
        print("self.data",self.data)
        y_predicted = self.predict_preprocessed(self.data)
        return y_predicted


    def _features_engineering(self):
        self.data = convert_df_to_numeric_df(self.data,["RowNumber", "CustomerId", "Surname","CustomerFeedback"])
        self.data["RatioSalary"] = self.data["EstimatedSalary"] /  (self.data["Balance (EUR)"]+1)
        self.data["RatioProducst"] = self.data["NumberOfProducts"] / (self.data["Tenure"] +1)
        self.data["RatioCards"] = self.data["HasCreditCard"] &  self.data["IsActiveMember"]
        self.data["ProductSalary"] = self.data["Balance (EUR)"] *  self.data["CreditScore"]
        self.scale_data()

    def interpret_shap(self):
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(self.data)
        return shap_values


    def interpret_bert(self):
        explainer = SequenceClassificationExplainer(self.bert_model.model,self.bert_model.tokenizer)
        corpus_by_feedback = [ nltk.sent_tokenize(text) if isinstance(text,str) else 0 for text in self.feedbacks ]
        attributions_by_feedback =[]
        for counter,feedback in enumerate(corpus_by_feedback) :
            by_feed  = []
            if feedback != 0 :
                for sentence in feedback :
                    attributions = explainer(sentence)
                    html = explainer.visualize()
                    by_feed.append(html)
                attributions_by_feedback.append(by_feed)
            else :
                attributions_by_feedback.append([0])
        return attributions_by_feedback

                



