# RobustWorld

## Uploading a video

Drop the `.mp4` into:

```
data/raw/videos/
```

Then commit and push:

```bash
cd ~/RobustWorld
git add data/raw/videos/<your-video>.mp4
git commit -m "Add capture video"
git push -u origin main
```

### Size limits (important)

GitHub **hard-rejects any single file over 100 MB** and warns above 50 MB.
Check before committing:

```bash
ls -lh data/raw/videos/
```

If the file is over 100 MB you must use Git LFS, which is not yet installed
on this machine:

```bash
brew install git-lfs          # requires Homebrew
git lfs install
git lfs track "*.mp4"
git add .gitattributes
```

## Layout

| Path                   | Purpose                              | Tracked? |
|------------------------|--------------------------------------|----------|
| `data/raw/videos/`     | Source video uploads                 | yes      |
| `data/processed/`      | Extracted frames, derived data       | no       |
| `data/interim/`        | Scratch intermediates                | no       |
| `src/robust_world/`    | Package source                       | yes      |
| `scripts/`             | Entry-point scripts                  | yes      |
| `configs/`             | Experiment configs                   | yes      |
| `checkpoints/`         | Model weights                        | no       |
| `outputs/`             | Renders, results                     | no       |
| `notebooks/`           | Exploration                          | yes      |
| `tests/`               | Tests                                | yes      |
