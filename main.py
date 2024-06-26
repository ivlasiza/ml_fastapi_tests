from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
class Item(BaseModel):
    text: str


class HistoryData:
    data: list


app = FastAPI()
classifier = pipeline("sentiment-analysis")
history = HistoryData()
history.data = list()


@app.get("/")
	@@ -20,4 +24,10 @@ def root():

@app.post("/predict/")
def predict(item: Item):
    history.data.append(item.text)
    return classifier(item.text)[0]


@app.get("/stats")
def get_statistic():
    return {len(history.data): history.data}
