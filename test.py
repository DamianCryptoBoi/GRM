import requests
import argparse
parser = argparse.ArgumentParser(description="Send a prompt to an endpoint.")
parser.add_argument("prompt", metavar="mode", type=str)
args = parser.parse_args()
gen_response = requests.post("http://localhost:8888/generate", data={
    "prompt": args.prompt,
}, timeout=600)

print(f"Prompt: {args.prompt}")
print(f"Response: {gen_response.text}")
