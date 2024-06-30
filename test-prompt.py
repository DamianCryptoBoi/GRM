import requests
import base64
import time
import os  # Import the os module for file operations

with open("./prompt.txt", "r") as f:
    prompts = f.readlines()

for prompt in prompts:
    print("-"*50)
    print("Prompt: ", prompt)
    count = 0
    while True:
        count += 1
        start_time = time.time()  # Record the start time before the request
        gen_response = requests.post("http://localhost:8888/generate/", data={
            "prompt": prompt,
            "steps": 30 * count
        }, timeout=600)

        end_time = time.time()  # Record the end time after the request
        generation_time = end_time - start_time  # Calculate the duration

        ply_file_path = gen_response.text.replace('"', '')

        with open(ply_file_path, 'rb') as file:
            file_content = base64.b64encode(file.read()).decode('utf-8')
        response = requests.post("http://localhost:8094/validate_ply/",
                                 json={"prompt": prompt, "data": file_content, "data_ver": 1})

        score = response.json().get("score", 0)

        print(f"Score: {score}")
        # Print the generation time
        print(f"Generation Time: {generation_time} seconds")

        os.remove(ply_file_path)  # Remove the PLY file after processing
        # Optional: Confirm file removal
        print(f"Removed file: {ply_file_path}")

        if score > 0.6:
            print("Success!, Took ", count, " tries")
            break
    print("-"*50)
