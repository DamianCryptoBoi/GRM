import requests
import time

with open("./prompt.txt", "r") as f:
    prompts = f.readlines()

for prompt in prompts:
    print("Prompt: ", prompt)
    gen_response = requests.post("http://localhost:8888/generate/", data={
        "prompt": prompt,
    }, timeout=600)
    time.sleep(1)
