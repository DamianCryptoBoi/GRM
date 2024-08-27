import requests
import time

with open("filtered_result.txt", "r") as textfile:
    prompts = [prompt.split(",")[0] for prompt in  textfile.readlines()]

pass_count = 0
total_runs = 0
start_time = time.time()
final_score = 0

def print_score():
    print("-"*50)
    print(f"Total runs: {total_runs}")
    print(f"Pass count: {pass_count}")
    print(f"Pass rate: {pass_count/total_runs*100}%")
    print(f"Final score: {final_score}")
    print("-"*50)

for prompt in prompts:

    total_runs += 1
    print(f"Prompt: {prompt.strip()}")
    try:
        gen_response = requests.post("http://localhost:8193/generate/", data={
            "prompt": prompt.strip(),
        }, timeout=600)
        score = float(gen_response.text)
        print(f"score: {score}")
        if score > 0.6:
            pass_count += 1
            final_score += 0.75
            if score > 0.8:
                final_score += 0.25
        else:
            final_score -= 2.5
        print_score()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
print_score()