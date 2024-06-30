import requests
import time
import argparse
parser = argparse.ArgumentParser(description="Send a prompt to an endpoint.")
parser.add_argument("prompt", metavar="mode", type=str)
args = parser.parse_args()
start_time = time()
gen_response = requests.post("http://localhost:8093/generate", data={
    "prompt": args.prompt,
}, timeout=600)

print(f"Prompt: {args.prompt}")
print(f"Response: {gen_response.text}")
print(f"Time taken: {time() - start_time}")
