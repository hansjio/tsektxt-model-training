"""Upload the hand-refined model cards in model_cards/ to the three Hugging Face repos.

    python src/upload_cards.py          (needs HF_TOKEN env var, or a prior `huggingface-cli login`)
"""
from pathlib import Path
from huggingface_hub import HfApi

REPOS = {"roberta-tagalog": "tsektxt-roberta-tagalog", "mbert": "tsektxt-mbert", "xlmr": "tsektxt-xlmr"}
if __name__ == "__main__":
    api = HfApi()
    for key, repo in REPOS.items():
        api.upload_file(path_or_fileobj=str(Path("model_cards") / f"{key}.md"), path_in_repo="README.md",
                        repo_id=f"hansjio/{repo}", commit_message="Refined model card with final results")
        print("updated", repo)
