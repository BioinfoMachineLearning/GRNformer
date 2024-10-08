import torch
import torch.nn.functional as F
import random

def sample_embedding_based_negatives(embeddings, edge_index, num_neg_samples, threshold=0.8):
    
    embeddings =embeddings.squeeze(0)
    num_nodes = embeddings.size(0)
    
    hard_negatives = []
    
    # Convert edge_index to a set for fast lookup
    edge_set = set(map(tuple, edge_index.t().tolist()))
    
    # Sample pairs of nodes
    while len(hard_negatives) < num_neg_samples:
        # Randomly sample two nodes
        node_a, node_b = random.randint(0, num_nodes-1), random.randint(0, num_nodes-1)
        
        # Skip if they already have an edge
        if (node_a, node_b) in edge_set or (node_b, node_a) in edge_set:
            continue
        
        # Calculate cosine similarity between their embeddings
        similarity = F.cosine_similarity(embeddings[node_a].unsqueeze(0), embeddings[node_b].unsqueeze(0))
        #print(similarity)
        # Select as hard negative if similarity is above the threshold
        if similarity.mean() > threshold:
            hard_negatives.append((node_a, node_b))
    
    return torch.tensor(hard_negatives).t()
