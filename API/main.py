from fastapi  import FastAPI
from pydantic import BaseModel
from preTrainedModel.bertGoEmotions import get_emotions

app = FastAPI()

class JournalEntry(BaseModel):
    text: str

@app.post("/predict/")
async def predict_emotions(journal_entry: JournalEntry):
    emotions = await get_emotions(journal_entry.text)
    return {"emotion_predictions": emotions}