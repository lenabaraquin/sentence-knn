import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel
import torch
from typing import Dict, List, Tuple

###############
# Configuration
###############

def build_parser():
    parser = argparse.ArgumentParser(
        description="Calcule les k plus proches voisins d'un embedding de phrase parmi les embeddings d'un corpus, selon la méthode de pooling spécifiée."
    )
    parser.add_argument("corpus_path", help="Chemin vers le fichier JSON du corpus.")
    parser.add_argument(
        "-p", "--pooling",
        help="Méthode de pooling pour l'embedding de phrase.",
        choices=["cls", "mean", "two_layer_mean", "2_layer_mean"],
        default="mean"
    )
    parser.add_argument(
        "-m", "--model",
        help="Checkpoint du modèle Hugging Face.",
        default="sentence-transformers/LaBSE"
    )
    #parser.add_argument(
        #"-q", "--query",
        #help="Requête.", 
    #)
    parser.add_argument(
        "-t", "--text_field",
        help="Nom du champ texte dans le corpus.",
        default="text"
    )
    parser.add_argument(
        "-k", "--mask_field",
        help="Nom du champ de masque d'attention si présent (optionnel, ignoré si absent).",
        default=None
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        help="Taille des batchs.",
        default=64
    )
    parser.add_argument(
        "-l", "--max_length",
        type=int,
        help="Longueur maximale (en tokens).",
        default=128
    )
    parser.add_argument(
        "--save_dir",
        help="Dossier de sauvegarde du dataset enrichi.",
        default="./corpus_with_embeddings.json"
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Nombre de plus proches voisins à retourner pour l'exemple de requête.",
        default=5
    )
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: str) -> Dict[str, object]:
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to(device).eval()
    return {"model": model, "tokenizer": tokenizer, "device": device}

###############
# Pooling
###############

@torch.no_grad()
def cls_pooling(last_hidden_state: torch.FloatTensor) -> torch.FloatTensor:
    """
    last_hidden_state: (batch_size, seq_len, hidden_size)
    return: (batch_size, hidden_size) — embedding du token CLS
    """
    return last_hidden_state[:, 0, :]

@torch.no_grad()
def mean_pooling(token_embeddings: torch.FloatTensor,
                 attention_mask: torch.LongTensor) -> torch.FloatTensor:
    """
    token_embeddings: (batch_size, seq_len, hidden_size)
    attention_mask:   (batch_size, seq_len)
    return: (batch_size, hidden_size) — moyenne masquée
    """
    mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)  # (B, L, 1)
    summed = (token_embeddings * mask).sum(dim=1)                  # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                       # (B, 1)
    return summed / counts

@torch.no_grad()
def two_layers_mean_pooling(hidden_states: Tuple[torch.FloatTensor, ...],
                            attention_mask: torch.LongTensor) -> torch.FloatTensor:
    """
    hidden_states: tuple de longueurs de couches, chaque (B, L, H).
    On moyenne les 2 dernières couches puis on applique mean pooling masqué.
    """
    last = hidden_states[-1]
    penultimate = hidden_states[-2]
    avg_last_two = 0.5 * (last + penultimate)
    return mean_pooling(avg_last_two, attention_mask)

###############
# Embeddings
###############

@torch.no_grad()
def get_embeddings(batched_texts: List[str],
                   bundle: Dict[str, object],
                   pooling: str,
                   max_length: int) -> torch.FloatTensor:
    """
    Renvoie un tenseur (batch_size, hidden_size) d'embeddings.
    """
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]

    norm_pooling = "two_layer_mean" if pooling == "2_layer_mean" else pooling
    need_hidden = (norm_pooling == "two_layer_mean")

    encoded = tokenizer(
        batched_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    outputs = model(**encoded, output_hidden_states=need_hidden)

    if norm_pooling == "cls":
        embs = cls_pooling(outputs.last_hidden_state)
    elif norm_pooling == "mean":
        embs = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
    elif norm_pooling == "two_layer_mean":
        assert outputs.hidden_states is not None, "hidden_states manquants"
        embs = two_layers_mean_pooling(outputs.hidden_states, encoded["attention_mask"])
    else:
        raise ValueError('pooling doit être "cls", "mean" ou "two_layer_mean"')

    return embs.detach().to("cpu")

###############
# Distance
###############

def chebyshev_knn(embeddings_batch: torch.FloatTensor,
                  query_embedding: torch.FloatTensor,
                  k: int):
    """
    embeddings_batch: (N, H)
    query_embedding:  (H,)
    Retourne (valeurs, indices) des k plus proches selon la distance de Tchebychev.
    """
    # Broadcasting: (N, H) - (H,) -> (N, H)
    distances = (embeddings_batch - query_embedding).abs().amax(dim=1)  # (N,)
    return torch.topk(distances, k, largest=False, sorted=True)

###############
# Post-traitement
###############

def from_id_return_sentence(ids: List[int], dataset: Dataset, text_field: str) -> str:
    return dataset.select(ids)[text_field]

###############
# Pipeline
###############

def main():
    args = build_parser()
    bundle = load_model(args.model)
    query = input("Entrez la requête : ")

    print("→ Chargement du dataset…")
    dataset = load_dataset("json", data_files=args.corpus_path, split="train")

    # Nettoyage simple: garder les exemples avec texte non vide et suffisamment long
    text_col = args.text_field
    if text_col not in dataset.column_names:
        raise KeyError(f"Colonne '{text_col}' introuvable dans le dataset. Colonnes: {dataset.column_names}")

    dataset = dataset.filter(lambda x: isinstance(x.get(text_col, ""), str) and len(x[text_col]) > 15)
    dataset = dataset.remove_columns("id")
    dataset = dataset.map(
        lambda samples, indices: {"id": indices},
        with_indices=True,
        batched=True
    )
    print("→ Dataset filtré.")

    # Si les embeddings existent déjà, on évite de recalculer
    emb_col = f"embeddings_{'two_layer_mean' if args.pooling in ('two_layer_mean','2_layer_mean') else args.pooling}"
    already_computed = (emb_col in dataset.column_names)
    if already_computed:
        print(f"→ La colonne '{emb_col}' existe déjà. Pas de recalcul.")

        embeddings_dataset = dataset
    else:
        print(f"→ Calcul des embeddings ({emb_col})…")

        def map_batch_to_embeddings(batch):
            texts = batch[text_col]
            embs = get_embeddings(
                texts,
                bundle=bundle,
                pooling=args.pooling,
                max_length=args.max_length
            )
            # Convertir en listes pour stockage Apache Arrow
            return {emb_col: embs.numpy().tolist()}

        embeddings_dataset = dataset.map(
            map_batch_to_embeddings,
            batched=True,
            batch_size=args.batch_size
        )
        print("→ Embeddings calculés.")

    # Sauvegarde sur disque (dossier)
    if not(already_computed):
        print(f"→ Sauvegarde du dataset enrichi dans: {args.save_dir}")
        embeddings_dataset.to_json(args.save_dir)

    # Démo KNN rapide: on convertit la colonne d'embeddings en tenseur
    print("→ Démo KNN (Tchebychev) sur une requête jouet…")
    embeddings_dataset.set_format(type=None)  # s'assurer de recevoir des listes Python
    embs_tensor = torch.tensor(embeddings_dataset[emb_col])  # (N, H)

    query_emb = get_embeddings(
        [query],
        bundle=bundle,
        pooling=args.pooling,
        max_length=args.max_length
    )[0]  # (H,)

    k = min(args.k, embs_tensor.size(0))
    values, indices = chebyshev_knn(embs_tensor, query_emb, k)
    
    knn_sentences = from_id_return_sentence(indices.tolist(), embeddings_dataset, args.text_field)
    for i in range(k):
        print(f"rang : {i}\t distance : {values[i]}\t indice : {indices[i]}\n\t {knn_sentences[i]}\n")
        
if __name__ == "__main__":
    main()
