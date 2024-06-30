from fastapi import FastAPI, Form
from processor import GRMProcessor
import uvicorn

processor = GRMProcessor()
app = FastAPI()


@app.get("/")
async def check():
    return {"status": "ok"}


@app.post("/generate")
async def generate(prompt: str = Form()):
    return processor.process(prompt)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
