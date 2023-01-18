import uvicorn
from fastapi import FastAPI, Request
import pandas as pd
import pickle
from pydantic import BaseModel


class Iris(BaseModel):
 sepal_length: float
 sepal_width: float
 petal_length: float
 petal_width: float

api = FastAPI()


@api.get("/")
async def read_root():
    return {"classification": "Iris"}

# Loading save model(.pkl)
pkl_filename = "iris_knn.pkl"
#pkl_filename = "iris_random.pkl"
#pkl_filename = "iris_svm.pkl"
#pkl_filename = "iris_logistic.pkl"
with open(pkl_filename, 'rb') as file:
 lr_model = pickle.load(file)


@api.post('/predict')
async def predict(iris: Iris):
    # Converting input data into Pandas DataFrame
    input_df = pd.DataFrame([iris.dict()])

    # Getting the prediction from the Logistic Regression model
    pred = lr_model.predict(input_df)[0]

    return pred

if __name__ == "__main__":
    uvicorn.run(api, host="localhost", port=8000)