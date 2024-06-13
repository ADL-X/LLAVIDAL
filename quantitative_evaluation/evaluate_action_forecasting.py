import openai
import argparse
import json
import ast

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--json_path", required=False, help="The path to the JSON file containing video data.",default="")
    parser.add_argument("--api_key", required=False, help="OpenAI API key.",default="")
    args = parser.parse_args()
    return args

def main():
    # Parse arguments.
    args = parse_args()

    # Read the JSON file
    with open(args.json_path) as file:
        video_data = json.load(file)

    video_results = video_data['results']    

    # Preparing dictionary of video data
    video_set = {}
    for sample in video_results:
        video_id = sample['video_id']
        ground_truth = sample['ground_truth']
        prediction = sample['prediction']
        video_set[video_id] = {"ground_truth": ground_truth, "prediction": prediction}

    # Set the OpenAI API key.
    openai.api_key = args.api_key

    # Process the video data
    results = {}
    for video_id, data in video_set.items():
        ground_truth = data['ground_truth']
        prediction = data['prediction']
        try:
            # Compute the similarity score using GPT-3
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": 
                            "You are an intelligent chatbot designed for evaluating the similarity between ground truth and predicted actions in videos. "
                            "Your task is to compare the predicted action with the ground truth action and determine if they match meaningfully. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the meaningful match between the predicted action and the ground truth action.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the similarity of the prediction compared to the ground truth."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based action pair:\n\n"
                            f"Ground Truth: {ground_truth}\n"
                            f"Predicted Action: {prediction}\n\n"
                            "Provide your evaluation only as a yes/no, where 'yes' indicates a meaningful match and 'no' indicates a mismatch. "
                            "Please generate the response in the form of a Python dictionary string with key 'similarity', where the value is a string of 'yes' or 'no'."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'similarity': 'yes'}."
                    }
                ]
            )
            # Convert response to a Python dictionary.
            response_message = completion["choices"][0]["message"]["content"]
            response_dict = ast.literal_eval(response_message)
            results[video_id] = response_dict['similarity']

        except Exception as e:
            print(f"Error processing video '{video_id}': {e}")

    # Calculate accuracy
    correct_count = sum(1 for result in results.values() if result == 'yes')
    total_count = len(results)
    accuracy = correct_count / total_count
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()