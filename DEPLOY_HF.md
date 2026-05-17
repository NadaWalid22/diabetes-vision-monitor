# Deploy to Hugging Face Spaces

Step-by-step to get a permanent public URL in ~10 minutes.

## 1. Create a Hugging Face account (if you don't have one)
https://huggingface.co/join

## 2. Install the HF CLI
```bash
pip install huggingface_hub
huggingface-cli login   # paste your HF token when prompted
```
Get your token at: https://huggingface.co/settings/tokens (write access)

## 3. Create the Space
```bash
huggingface-cli repo create glucosense-ai --type space --space_sdk streamlit
```

## 4. Push this repo to the Space
```bash
cd /path/to/diabetes-vision-monitor

git init
git add .
git commit -m "Initial deploy"

git remote add space https://huggingface.co/spaces/NadaWalid22/glucosense-ai
git push space main
```
## 5. Update the README link
Edit README.md line:
```
**Live demo:** [glucosense-ai on Hugging Face Spaces](https://huggingface.co/spaces/NadaWalid22/glucosense-ai)
```
## Notes
- First build takes 3-5 minutes (installing torch)
- The Space auto-rebuilds on every push
- Free CPU tier is fine for this demo (torch is installed as CPU-only via requirements.txt)
- If the Space goes to sleep (after inactivity), it wakes automatically when someone visits
