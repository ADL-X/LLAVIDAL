import openai
import os
import json
import ast
import argparse
import warnings
from tqdm import tqdm

# Suppressing all warnings
warnings.filterwarnings('ignore')

def parse_args():
    """
    Command-line argument parser.
    """
    parser = argparse.ArgumentParser(description="Descriptive question-answer-generation-using-GPT-3")
    parser.add_argument("--step3_description_json", required=True, help="Path to the image captions from step3.")
    parser.add_argument("--openai_api_key", required=True, help="OpenAI API key.")
    return parser.parse_args()

def annotate(captions):
    """
    Generates question and answer pairs based on video captions using OpenAI GPT-3 and returns the response dictionary.
    """
    mega_caption = " ".join(captions) + "\n"
    
    # Generate completion with OpenAI GPT-3
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": ("You will play two roles: a human asking questions related to describing a video and "
                                "an intelligent chatbot designed for video description and dense captioning. "
                                "Your task is to generate a detailed and descriptive paragraph based on the provided fragmented information about a video. "
                                "------"
                                "##TASK:"
                                "Users will provide fragmented descriptions of a video, and you will generate ONE conversation-like question and answer related to describing the video in detail. "
                                "The question should ask to describe the video content in detail. "
                                "The answer should be a paraphrased and well-structured paragraph based on the provided description, with a minimum of 150 words and a maximum of 300 words. "
                                "When the provided information is short, aim for a 150-word description, and when the provided information is more detailed, aim for very long descriptions up to 300-word description. "
                                "------"
                                "##INSTRUCTIONS:"
                                "- The question must be like a human conversation and focused on describing the video in detail. "
                                "- The answer must be a paraphrased version of the provided information, very detailed and descriptive, and within the specified word count. "
                                "- Combine the information from different sections of the video into a single coherent summary, ignoring any repetitions."
                                "- Compare the information across all fragments of video and remove or ignore any inconsistent information and do not say the summary comes from different fragments of the video."
                                "- Give more emphasis on the actions, the objects, and the colors of the background and the objects."
                                "- Give the sequence of actions happening in the video and the objects the person interacts with.")
                },
                {
                    "role": "user",
                    "content": f"The fragmented video description is: {mega_caption}. "
                               '''Please generate the response in the form of a Python dictionary string with keys "Q" for question and "A" for answer. Each corresponding value should be the question and answer text respectively.'''
                               '''For example, your response should look like this: {"Q": "Your question here...", "A": "Your answer here..."}.'''
                               '''Emphasize that the answer should focus on describing the video content following the given instructions.'''
                }
            ],
        )
        try:
            response_data = completion.choices[0].message.content
            response_dict = ast.literal_eval(response_data)
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing response data: {e}. Skipping this item.")
            response_dict = {"error": "Failed to parse response data."}
    except openai.error.RateLimitError as e:
        print(f"OpenAI API rate limit exceeded: {e}. Skipping this item.")
        response_dict = {"error": "OpenAI API rate limit exceeded."}
    except Exception as e:
        print(f"An error occurred: {e}. Skipping this item.")
        response_dict = {"error": "An unknown error occurred."}
    
    return response_dict

def main():
    args = parse_args()
    
    combined_output_path = './dense_descriptions.json'
    
    # Check if dense_descriptions.json already exists
    if os.path.exists(combined_output_path):
        print(f"dense_descriptions.json already exists at {combined_output_path}. Skipping processing.")
        return
    
    with open(args.step3_description_json) as file:
        image_captions = json.load(file)
    
    openai.api_key = args.openai_api_key  # Replace with your actual API key
    
    combined_annotations = {}
    save_interval = 10  # Save progress every 10 videos
    
    for idx, (video_id, descriptions) in enumerate(tqdm(image_captions.items())):
        response_dict = annotate(descriptions)
        combined_annotations[video_id] = response_dict
        
        # Save progress periodically
        if (idx + 1) % save_interval == 0 or (idx + 1) == len(image_captions):
            with open(combined_output_path, "w") as f:
                json.dump(combined_annotations, f, indent=4)

    print(f"Completed, all annotations saved in {combined_output_path}")

if __name__ == "__main__":
    main()