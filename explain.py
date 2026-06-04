import os
import argparse
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Import framework configurations and builders
from config.configurator import configs
from trainer.utils import set_seed
from models.bulid_model import build_model
from load_data.build_data_handler import build_data_handler
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="LLM-AGR Generative Explainability Module")
    parser.add_argument("--checkpoint", type=str, default=None, required=True,
                        help="Path to the saved model checkpoint file (.pth)")
    parser.add_argument("--num_users", type=int, default=5,
                        help="Number of users to generate explanations for")
    parser.add_argument("--output", type=str, default="results/explainability_report.md",
                        help="Output path for the markdown report")
    
    # Register core configuration CLI flags to avoid unknown arg errors
    parser.add_argument("--model", type=str, default="lightgcn_agr")
    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--diverse", type=int, default=2)
    parser.add_argument("--ablation", type=str, default="none")
    parser.add_argument("--attack_config", type=str, default=None)
    
    return parser.parse_known_args()[0]

def load_profiles(dataset):
    print(f"Loading user and item profiles for dataset: {dataset}...")
    user_file = f"./data/{dataset}/usr_prf.pkl"
    item_file = f"./data/{dataset}/itm_prf.pkl"
    
    with open(user_file, 'rb') as f:
        usr_prf = pickle.load(f)
    with open(item_file, 'rb') as f:
        itm_prf = pickle.load(f)
        
    return usr_prf, itm_prf

def get_nearest_items(proj_emb, item_gt, k=3):
    """
    Given a projected embedding (shape: [D]), find the top-k nearest items
    in the ground-truth item semantic space (shape: [M, D]).
    """
    proj_norm = F.normalize(proj_emb.view(1, -1), p=2, dim=-1)
    item_gt_norm = F.normalize(item_gt, p=2, dim=-1)
    cos_sim = torch.mm(proj_norm, item_gt_norm.T).view(-1)
    topk_vals, topk_indices = torch.topk(cos_sim, k=k)
    return topk_indices.cpu().tolist(), topk_vals.cpu().tolist()

def generate_explanation(user_profile, item_profile, user_neighbors, item_neighbors, model, tokenizer):
    """
    Constructs prompt and uses Qwen-0.5B-Instruct to generate natural language explanation.
    """
    prompt = f"""You are a recommendation explainability assistant. 
Based on the user's history and profiles, explain why the recommended item is a good match for the user.

User Profile: {user_profile}
User's Graph-learned semantic interests: {user_neighbors}
Recommended Item: {item_profile}
Recommended Item's Graph-learned semantic features: {item_neighbors}

Write a short, engaging, and personalized 2-3 sentence explanation explaining why the recommended item fits the user's preferences:"""

    messages = [
        {"role": "system", "content": "You are a helpful and concise recommendation assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def main():
    args = parse_args()
    set_seed(configs['train']['seed'])
    
    print("Loading data...")
    data_handler = build_data_handler()
    data_handler.load_data()
    
    # Load model and weights
    print(f"Building model {configs['model']['name']}...")
    model = build_model(data_handler).to(configs['device'])
    print(f"Loading checkpoint weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=configs['device'])
    model.load_state_dict(checkpoint)
    model.eval()
    
    usr_prf, itm_prf = load_profiles(configs['data']['name'])
    
    # ---------------------------------------------------------
    # 1. Quantitative Semantic Alignment Analysis
    # ---------------------------------------------------------
    print("\nCalculating quantitative semantic alignment metrics...")
    
    # Extract CF embeddings from model backbone
    if hasattr(model, 'user_embeds'):
        user_cf = model.user_embeds
        item_cf = model.item_embeds
    elif hasattr(model, 'user_embedding'):
        user_cf = model.user_embedding.weight
        item_cf = model.item_embedding.weight
    else:
        raise AttributeError("Could not locate user/item embeddings in model backbone.")
        
    user_gt = model.usrprf_embeds
    item_gt = model.itmprf_embeds
    
    with torch.no_grad():
        # Project CF embeddings to semantic space via trained projector
        user_proj = model.gen_mlp(user_cf)
        item_proj = model.gen_mlp(item_cf)
        
        # Calculate cosine similarities
        user_cos = F.cosine_similarity(user_proj, user_gt, dim=1)
        item_cos = F.cosine_similarity(item_proj, item_gt, dim=1)
        
    mean_user_cos = user_cos.mean().item()
    std_user_cos = user_cos.std().item()
    mean_item_cos = item_cos.mean().item()
    std_item_cos = item_cos.std().item()
    
    print(f"User Semantic Alignment (Projected vs GT): {mean_user_cos:.4f} ± {std_user_cos:.4f}")
    print(f"Item Semantic Alignment (Projected vs GT): {mean_item_cos:.4f} ± {std_item_cos:.4f}")
    
    # ---------------------------------------------------------
    # 2. Recommendation Prediction and Explanations
    # ---------------------------------------------------------
    llm_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\nLoading {llm_name} generator...")
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype="auto",
        device_map="auto"
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
    
    print("Selecting test users and generating recommendations...")
    test_dataloader = data_handler.test_dataloader
    
    all_test_users = test_dataloader.dataset.test_users
    # Pick random test users
    rng = np.random.default_rng(configs['train']['seed'])
    sampled_users = rng.choice(all_test_users, size=min(args.num_users, len(all_test_users)), replace=False).tolist()
    
    # Gather historical interaction masks for predictions
    user_pos_lists = test_dataloader.dataset.user_pos_lists
    csrmat = test_dataloader.dataset.csrmat
    
    explanations_report = []
    
    for u in tqdm(sampled_users, desc="Generating explanations"):
        # Predict ratings for this user
        u_tensor = torch.LongTensor([u]).to(configs['device'])
        pck_mask = csrmat[u].toarray().reshape(-1)
        mask_tensor = torch.FloatTensor(pck_mask).to(configs['device']).view(1, -1)
        
        with torch.no_grad():
            batch_pred = model.full_predict([u_tensor, mask_tensor])
            _, topk_items = torch.topk(batch_pred, k=1)
            recommended_item = topk_items[0, 0].item()
            
            # Project embeddings for this user-item pair
            u_proj = user_proj[u]
            i_proj = item_proj[recommended_item]
            
            # Retrieve nearest neighbors in LLM semantic space
            user_nn_indices, _ = get_nearest_items(u_proj, item_gt, k=2)
            item_nn_indices, _ = get_nearest_items(i_proj, item_gt, k=2)
            
        # Get profile texts
        u_profile = usr_prf.get(u, {}).get("profile", "No profile description available.")
        i_profile = itm_prf.get(recommended_item, {}).get("profile", "No item description available.")
        
        user_nn_texts = [itm_prf.get(idx, {}).get("profile", "Unknown item") for idx in user_nn_indices]
        item_nn_texts = [itm_prf.get(idx, {}).get("profile", "Unknown item") for idx in item_nn_indices]
        
        # Keep only the first sentence of neighbors for conciseness in prompt
        u_nn_summary = " and ".join([text.split('.')[0] + '.' for text in user_nn_texts])
        i_nn_summary = " and ".join([text.split('.')[0] + '.' for text in item_nn_texts])
        
        # Generate explanation using Qwen
        explanation = generate_explanation(
            user_profile=u_profile,
            item_profile=i_profile,
            user_neighbors=u_nn_summary,
            item_neighbors=i_nn_summary,
            model=llm_model,
            tokenizer=llm_tokenizer
        )
        
        explanations_report.append({
            "user_id": u,
            "user_profile": u_profile,
            "item_id": recommended_item,
            "item_profile": i_profile,
            "user_neighbors": u_nn_summary,
            "item_neighbors": i_nn_summary,
            "explanation": explanation
        })
        
    # Write report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("# LLM-AGR Generative Explainability Report\n\n")
        f.write("## Quantitative Semantic Alignment Analysis\n\n")
        f.write("This section measures the cosine similarity between the MLP-projected GNN collaborative embeddings and their corresponding ground-truth LLM semantic profile embeddings.\n\n")
        f.write("| Representation | Cosine Similarity (Projected vs Ground Truth) |\n")
        f.write("| --- | --- |\n")
        f.write(f"| User Embeddings | {mean_user_cos:.4f} ± {std_user_cos:.4f} |\n")
        f.write(f"| Item Embeddings | {mean_item_cos:.4f} ± {std_item_cos:.4f} |\n\n")
        
        f.write("## Generated Recommendations & Explanations\n\n")
        f.write("Explanations generated using a local **Qwen2.5-7B-Instruct** model, guided by the projected graph representations.\n\n")
        
        for idx, item in enumerate(explanations_report):
            f.write(f"### Example {idx + 1}: User {item['user_id']} → Item {item['item_id']}\n\n")
            f.write(f"**User Profile:**\n> {item['user_profile']}\n\n")
            f.write(f"**Recommended Item Profile:**\n> {item['item_profile']}\n\n")
            f.write(f"**Graph-Projected User Semantic Neighbors:**\n> {item['user_neighbors']}\n\n")
            f.write(f"**Graph-Projected Item Semantic Neighbors:**\n> {item['item_neighbors']}\n\n")
            f.write(f"**Generated Explanation:**\n> **{item['explanation']}**\n\n")
            f.write("---\n\n")
            
    print(f"\nExplainability report successfully written to {args.output}!")

if __name__ == "__main__":
    main()
