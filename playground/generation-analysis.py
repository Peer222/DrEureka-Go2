from typing import Union, Optional, Tuple, List, Literal, Dict
from pathlib import Path
import pandas as pd
import numpy as np
import ast
import re
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.metrics.pairwise import cosine_similarity

from plots_plus.plot import clusterplot
from plots_plus.colors import LLM_COLOR_MAP


@dataclass
class Args:
    statspath: Path
    """Path to stats.csv file from eureka run"""
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    """Name of the huggingface embedding model name"""
    models_dir: Path = Path("/bigwork/nhwpduep/master_thesis/models/")
    """Path to model root directory"""
    prompt: Optional[str] = None
    """Not adjustable! Prompt that is prepended for embedding generation"""
    seed: int = 0


def load_model(model_name: Union[str, Path]) -> SentenceTransformer:
    return SentenceTransformer(str(model_name))


def get_reward_names(
    stats_df: pd.DataFrame,
) -> Tuple[List[str], List[Dict], List[str], List[int]]:
    """_summary_

    Args:
        stats_df (pd.DataFrame): _description_

    Returns:
        Tuple[List[str], Dict[str, List[str]], List[str], List[int]]: all names, names per iteration, best result names, counts
    """
    stats_df["reward_names"] = stats_df["reward_names"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    all = pd.Series([n for names in stats_df["reward_names"].values for n in names])
    counts = all.value_counts(sort=False).tolist()
    all = all.drop_duplicates().tolist()
    per_iteration = []
    for i, iteration_df in stats_df.groupby("iteration"):
        names = pd.Series(
            [n for names in iteration_df["reward_names"].values for n in names]
        )
        per_iteration.append(
            {
                "iteration": i,
                "counts": names.value_counts(sort=False).tolist(),
                "reward_names": names.drop_duplicates(),
            }
        )
    best_idx = np.argmax(stats_df["fitness_score_max"])
    best = stats_df.iloc[best_idx]["reward_names"]
    return all, per_iteration, best, counts


def get_rewards(reward_dir: Path) -> pd.DataFrame:
    reward_files = reward_dir.glob("*.py")
    rewards = []
    for reward_file in reward_files:
        match = re.match(r"[^\d]*(\d+)[^\d]*(\d+)", reward_file.stem)
        if match is None:
            raise Exception(f"Unknown file pattern found: {reward_file.stem}")
        match.group(1)  # iteration
        with open(reward_file, "r") as f:
            content = f.read()

        rewards.append(
            {
                "iteration": int(match.group(1)),
                "sample": int(match.group(2)),
                "content": content,
            }
        )
    rewards_df: pd.DataFrame = pd.DataFrame(rewards).sort_values(
        ["iteration", "sample"]
    )
    rewards_df.reset_index(inplace=True)
    return rewards_df


def get_mds(
    texts: List[str], embeddings: np.ndarray, method: Literal["PCA", "t-SNE", "spectral"]
) -> pd.DataFrame:
    if method == "PCA":
        solver = PCA(n_components=2)
        transformed_embeddings = solver.fit_transform(embeddings)
        embedding_df = pd.DataFrame(
            transformed_embeddings, columns=["1. Component", "2. Component"]
        )
    elif method == "t-SNE":
        solver = TSNE(n_components=2, metric="cosine")
        transformed_embeddings = solver.fit_transform(embeddings)
        embedding_df = pd.DataFrame(transformed_embeddings, columns=["x", "y"])
    elif method == "spectral":
        solver = SpectralEmbedding(n_components=2, affinity=cosine_similarity)
        transformed_embeddings = solver.fit_transform(embeddings)
        embedding_df = pd.DataFrame(transformed_embeddings, columns=["x", "y"])
    embedding_df["text"] = texts
    return embedding_df


def compute_mds_from_text(
    args: Args, texts: List[str], text_type: str, method: Literal["PCA", "t-SNE"]
) -> pd.DataFrame:
    embedding_path = (
        args.statspath.parent
        / "embeddings"
        / f"{text_type}_{args.embedding_model.split('/')[-1]}.npy"
    )
    if (embedding_path).exists():
        embeddings = np.load(embedding_path)
        print(f"Loaded embeddings from {embedding_path}")
        mds_df = get_mds(texts, embeddings, method)
    else:
        model = load_model(args.models_dir / args.embedding_model)
        embeddings = model.encode(
            texts, prompt=args.prompt, show_progress_bar=True, batch_size=20
        )
        (args.statspath.parent / "embeddings").mkdir(exist_ok=True)
        np.save(embedding_path, embeddings)

        mds_df = get_mds(texts, embeddings, method)
    if text_type == "rewards":
        mds_df = mds_df.rename({"text": "code"}, axis=1)
    return mds_df


if __name__ == "__main__":
    import tyro

    args = tyro.cli(Args)
    np.random.seed(args.seed)
    stats_df = pd.read_csv(args.statspath)

    version_order = []
    match = re.search(".*/([^_]+)_.*", stats_df["version"].iloc[0])
    if match is None:
        raise Exception(f"Unknown model version found: {stats_df['version'].iloc[0]}")
    version_order.append(match.group(1))
    stats_df["version"] = match.group(1)

    ### rewards
    rewards_df = get_rewards(args.statspath.parent / "rewards")
    args.prompt = ""  # "Instruct: Given a reward definition, retrieve relevant passages that answer the query\nQuery:"
    # PCA
    pca_df = compute_mds_from_text(
        args, rewards_df["content"].tolist(), "rewards", method="PCA"
    )
    pca_df["version"] = stats_df["version"].iloc[0]
    pca_df["iteration"] = rewards_df["iteration"]
    pca_df["sample"] = rewards_df["sample"]
    pca_df["fitness_score"] = stats_df["fitness_score_max"]
    clusterplot(
        pca_df,
        "1. Component",
        "2. Component",
        "iteration",
        size="fitness_score",
        alpha=0.75,
        filepath=args.statspath.parent
        / "graphics"
        / "gen_analysis"
        / "rewards_pca.png",
    )

    # t-SNE
    tsne_df = compute_mds_from_text(
        args, rewards_df["content"].tolist(), "rewards", method="t-SNE"
    )
    tsne_df["version"] = stats_df["version"].iloc[0]
    tsne_df["iteration"] = rewards_df["iteration"]
    tsne_df["sample"] = rewards_df["sample"]
    tsne_df["fitness_score"] = stats_df["fitness_score_max"]

    clusterplot(
        tsne_df,
        "x",
        "y",
        "iteration",
        size="fitness_score",
        alpha=0.75,
        filepath=args.statspath.parent
        / "graphics"
        / "gen_analysis"
        / "rewards_t-sne.png",
    )

    # Spectral Embedding
    spectral_df = compute_mds_from_text(
        args, rewards_df["content"].tolist(), "rewards", method="spectral"
    )
    spectral_df["version"] = stats_df["version"].iloc[0]
    spectral_df["iteration"] = rewards_df["iteration"]
    spectral_df["sample"] = rewards_df["sample"]
    spectral_df["fitness_score"] = stats_df["fitness_score_max"]

    clusterplot(
        spectral_df,
        "x",
        "y",
        "iteration",
        size="fitness_score",
        alpha=0.75,
        filepath=args.statspath.parent
        / "graphics"
        / "gen_analysis"
        / "rewards_spectral.png",
    )


    ### reward names
    all_names, names_per_iteration, best_names, counts = get_reward_names(stats_df)

    args.prompt = "Reward Component: "
    # PCA
    pca_df = compute_mds_from_text(args, all_names, "reward_names", "PCA")

    pca_df["version"] = stats_df["version"].iloc[0]
    pca_df["group"] = "all"
    pca_df["count"] = counts
    for best_name in best_names:
        idx = all_names.index(best_name)
        pca_df.loc[idx, "group"] = "best"

    clusterplot(
        pca_df,
        "1. Component",
        "2. Component",
        "group",
        hue_order=["all", "best"],
        size="count",
        colorpalette=LLM_COLOR_MAP,
        alpha=0.75,
        filepath=args.statspath.parent
        / "graphics"
        / "gen_analysis"
        / "reward_names_pca.png",
    )

    # t-SNE
    tsne_df = compute_mds_from_text(args, all_names, "reward_names", "t-SNE")

    tsne_df["version"] = stats_df["version"].iloc[0]
    tsne_df["group"] = "all"
    tsne_df["count"] = counts
    for best_name in best_names:
        idx = all_names.index(best_name)
        tsne_df.loc[idx, "group"] = "best"

    clusterplot(
        tsne_df,
        "x",
        "y",
        "group",
        hue_order=["all", "best"],
        size="count",
        colorpalette=LLM_COLOR_MAP,
        alpha=0.75,
        filepath=args.statspath.parent
        / "graphics"
        / "gen_analysis"
        / "reward_names_t-sne.png",
    )

    # Spectral Embedding
    spectral_df = compute_mds_from_text(args, all_names, "reward_names", "spectral")

    spectral_df["version"] = stats_df["version"].iloc[0]
    spectral_df["group"] = "all"
    spectral_df["count"] = counts
    for best_name in best_names:
        idx = all_names.index(best_name)
        spectral_df.loc[idx, "group"] = "best"

    clusterplot(
        spectral_df,
        "x",
        "y",
        "group",
        hue_order=["all", "best"],
        size="count",
        colorpalette=LLM_COLOR_MAP,
        alpha=0.75,
        filepath=args.statspath.parent
        / "graphics"
        / "gen_analysis"
        / "reward_names_spectral.png",
    )
