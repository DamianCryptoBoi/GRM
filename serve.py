from fastapi import FastAPI, Form
from processor import GRMProcessor

app = FastAPI()
processor = GRMProcessor()


@app.get("/")
async def check():
    return {"status": "ok"}


@app.post("/generate")
async def generate(prompt: str = Form()):
    return processor.generate(prompt)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port="8888")
