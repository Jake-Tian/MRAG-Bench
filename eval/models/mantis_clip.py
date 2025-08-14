import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import sys

# Import your mantis CLIP Llama3 model
import mantis
# Example: from mantis.clip_llama3 import load_model

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.dataloader import bench_data_loader

def eval_model(args):
    ans_file = open(args.answers_file, "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Mantis CLIP Llama3 model
    # Adjust the model loading as per your mantis API
    model, preprocess, tokenizer = mantis.load_clip_llama3(args.model_path, device=device)
    model.eval()

    for item in bench_data_loader(args, image_placeholder=None):
        qs = item['question']
        image_paths = item['image_files']
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        # Process images
        images = [
            preprocess(img_path.convert("RGB") if isinstance(img_path, Image.Image) else Image.open(img_path).convert("RGB")).unsqueeze(0)
            for img_path in image_paths
        ]
        images = torch.cat(images, dim=0).to(device)

        # Tokenize question for CLIP Llama3, with truncation if supported
        # If tokenizer has a truncate argument, use it; otherwise, truncate manually
        try:
            text = tokenizer([qs], truncate=True).to(device)
        except TypeError:
            # Fallback: truncate string to 77 chars (CLIP's context length)
            text = tokenizer([qs[:77]]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze().cpu().tolist()

        # If multiple images, keep the max similarity
        if isinstance(similarity, list):
            max_sim = max(similarity)
        else:
            max_sim = similarity

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "qs_id": item['id'],
            "prompt": item['prompt'],
            "output": f"Mantis CLIP Llama3 similarity: {max_sim:.4f}",
            "gt_answer": item['answer'],
            "shortuuid": ans_id,
            "model_id": 'mantis_clip_llama3',
            "gt_choice": item.get('gt_choice', None),
            "scenario": item.get('scenario', None),
            "aspect": item.get('aspect', None),
        }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="mantis/clip-llama3")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--use_rag", type=lambda x: x.lower() == 'true', default=False, help="Use RAG")
    parser.add_argument("--use_retrieved_examples", type=lambda x: x.lower() == 'true', default=False, help="Use retrieved examples")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)