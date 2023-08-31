import requests 
import pandas as pd

if __name__ == "__main__" :
    xlxs_file_path = '../data/assignement.xlsx' 
    df = pd.read_excel(xlxs_file_path)[-10 :]
    target = df["Exited"]
    df = df.drop(columns=["Exited"])
    data = {'file': ('data.csv', df.to_csv(index=False), 'text/csv')}
    url = 'http://127.0.0.1:5000/predict'
    response = requests.post(url, files=data).json()
    print(response.get("predictions"))
