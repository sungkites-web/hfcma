# GitHub Release Checklist

Before pushing this repository to GitHub, check the following items.

## Keep

- `src/`, `configs/`, `scripts/`, `docs/`, `README.md`, `requirements.txt`, `pyproject.toml`.
- Data preparation instructions and JSON schema.
- Reproducible command examples.
- Core model, dataset loader, training entry point, and evaluation entry point.

## Exclude

- Raw Flickr30K and MS-COCO images.
- Annotation files if their redistribution terms are unclear.
- Feature tensors, model checkpoints, caches, logs, spreadsheets, and temporary shell queues.
- Server IP addresses, usernames, passwords, local machine paths, and SSH commands.
- Temporary job managers, debug scripts, and machine-specific experiment queues.

## Suggested Git Commands

```bash
git init
git add README.md requirements.txt pyproject.toml .gitignore configs docs scripts src
git status
git commit -m "Release HFCMA reproducibility code"
```

After the final GitHub URL is available, update the README citation and repository link.

