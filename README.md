# AutoCode Analyzer

An AI-powered tool that takes a GitHub repository URL, indexes the codebase
with a Retrieval-Augmented Generation (RAG) pipeline, and answers natural-
language questions about what the code does -- citing the exact files it used.

Built end-to-end: local RAG pipeline -> Flask web app -> Docker container ->
CI/CD pipeline -> AWS deployment (ECR + EC2).

Runs entirely on free tiers: local ONNX embeddings (no PyTorch needed) and 
Groq's free API tier for the LLM. No OpenAI key, no paid vector database. 
Fits Render's free tier (512 MB container).

## How it works

```
 GitHub URL (any form: repo link, /tree/... link, .git, SSH)
     |
     v
 [1] Normalize URL + shallow clone (GitPython, non-interactive)
     |
     v
 [2] Load source files -- skip .git, node_modules, venvs, build output,
     lockfiles, minified bundles, binaries, oversized files
     |
     v
 [3] Language-aware chunking (one splitter per file type: Python, JS,
     Java, Go, ...), numbered per file
     |
     v
 [4] Embed chunks locally (ONNX all-MiniLM-L6-v2, CPU, free, ~80 MB)
     |
     v
 [5] Store in a Chroma collection (one per repo, rebuilt cleanly on re-analysis)
     |
     v
 [6] Question -> condense with chat history -> retrieve top-k chunks ->
     Groq LLM answers from that context only -> cites source files
```

Step 6 is a plain LangChain Expression Language (LCEL) chain
(`prompt | llm | parser`) rather than a black-box chain class, so every
step is easy to read and swap out.

## Tech stack

| Layer          | Choice                                                   |
|----------------|-----------------------------------------------------------|
| Language       | Python 3.10+ (tested on 3.11 and 3.12)                     |
| RAG framework  | LangChain / LangChain Expression Language (LCEL)           |
| LLM            | Groq API (`openai/gpt-oss-120b`) -- free tier               |
| Embeddings     | ONNX `all-MiniLM-L6-v2` (Chroma) -- local, no PyTorch (~80 MB)   |
| Vector store   | Chroma (embedded, on-disk)                                  |
| Repo access    | GitPython                                                    |
| Web app        | Flask + vanilla JS chat UI (safe Markdown rendering)          |
| Tests          | pytest -- mock Groq server + fake embeddings, no key needed   |
| Container      | Docker (CPU-only, ONNX embedded model baked in, ~500 MB)       |
| CI/CD          | GitHub Actions -> AWS ECR -> self-hosted runner on EC2 or Render  |

## Project structure

```
.
├── .github/workflows/cicd.yaml   # CI (tests) -> build & push to ECR -> deploy on EC2
├── notebook/experiments.ipynb    # step-by-step pipeline walkthrough
├── src/codeanalyzer/
│   ├── config.py                 # all settings, loaded from .env (project-root relative)
│   ├── logger.py, exceptions.py  # logging + errors with user-safe messages
│   ├── data_ingestion/           # URL parsing, cloning, file loading & filtering
│   ├── text_splitter/            # language-aware chunking
│   ├── embeddings/               # local HuggingFace embedding model
│   ├── vectorstore/              # Chroma wrapper (clean rebuilds, stable chunk IDs)
│   ├── llm/                      # Groq LLM client + API-key checks
│   ├── qa/                       # the LCEL RAG chain
│   └── pipeline.py               # ties it all together (thread-safe)
├── templates/, static/           # Flask chat UI
├── tests/                        # pytest suite (conftest has the mock Groq server)
├── app.py                        # Flask routes: /, /health, /status, /ingest, /chat
├── check_setup.py                # diagnoses .env / key / git / model problems
├── template.py                   # regenerates this folder skeleton
├── Dockerfile, .dockerignore
├── requirements.txt, requirements-dev.txt, setup.py
└── .env.example
```

## Local setup (Windows PowerShell)

**Prerequisites:** Python 3.10+ and Git.

```powershell
git clone https://github.com/Tharun5199/AI-END-TO-END-CODE-ANALYSIS.git
cd AI-END-TO-END-CODE-ANALYSIS

python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If PowerShell blocks the script:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install -r requirements.txt
pip install -e .
```

macOS / Linux: `source .venv/bin/activate` instead of the `Activate.ps1` line.

### 1. Create your `.env` and add your Groq key

```powershell
Copy-Item .env.example .env
notepad .env
```

Get a free key at <https://console.groq.com/keys> (it starts with `gsk_`) and
set `GROQ_API_KEY=gsk_...` in `.env`. Put the key **only** in `.env` --
`.env.example` is committed to GitHub, `.env` is not.

> Common pitfall: Notepad sometimes saves the file as `.env.txt`. The app
> detects this and tells you, but the fix is to rename it to exactly `.env`.

### 2. Check everything in one command

```powershell
python check_setup.py
```

It verifies Python, packages, git, your `.env` and key, GitHub access, the
embedding model download, Chroma, and makes one real call to Groq. Every
problem is printed with the exact fix.

### 3. Run the app

```powershell
python app.py
```

Open <http://localhost:8080>, paste a public GitHub repo URL, click
**Analyze repo**, then ask questions. The first run downloads the ~80 MB
ONNX embedding model once; it's cached afterwards. If the key isn't configured, a
yellow banner at the top of the page says exactly what's wrong.

### Notebook walkthrough

`notebook/experiments.ipynb` runs the same pipeline one stage at a time with
printed output (clone -> load -> split -> embed -> store -> ask), which is
the best way to understand -- and explain in an interview -- what each stage does.

### Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

No API key or model download needed: `tests/conftest.py` starts a local mock
of the Groq API so the real `ChatGroq` client is exercised end to end, and
embeddings are faked. The suite includes a regression test for every bug in
the "Troubleshooting" section below.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `GROQ_API_KEY is not set` | No `.env` file, or the key isn't in it. See step 1, then **restart** `python app.py`. |
| `GROQ_API_KEY still contains the placeholder text` | You copied `.env.example` but didn't replace `your_groq_api_key_here`. |
| `Groq rejected your API key (401)` | Key was mistyped or revoked -- create a new one at console.groq.com/keys. |
| `destination path ... already exists and is not an empty directory` | Fixed: git marks files read-only and Windows refused to delete them. Pull the latest code. |
| Answers repeat the same file 5 times | Fixed: re-analyzing a repo used to append duplicate chunks. Pull the latest code; the next analysis rebuilds the index cleanly. |
| `repository doesn't exist or is private` | Only public repos can be analyzed. Check the URL. |
| First "Analyze" is slow | The ONNX embedding model downloads once (~80 MB). Later runs are fast. |

## Running in Docker

```bash
docker build -t autocode-analyzer .
docker run -p 8080:8080 -e GROQ_API_KEY=gsk_your_key autocode-analyzer
```

The image uses local ONNX embeddings (no PyTorch) and has the embedding model 
baked in, so the container never needs to download it. Total image size: 
**~500 MB**, fits free-tier containers (Render, Railway, etc.). It runs gunicorn 
with **one** worker process and 4 threads on purpose: the analyzed repo lives 
in that process's memory, and multiple worker processes would each have their own copy.

## Free hosting on Render

Render's **free tier** includes:
- **512 MB RAM** + **0.1 vCPU** (CPU-only embeddings fit easily)
- **Deploys from GitHub** (push = instant redeploy)
- **Automatic redeploys** when you push to `main`
- **750 free instance hours/month** (~1 persistent instance)
- No credit card required
- Auto-spins down after 15 min inactivity (re-spins up when traffic returns, ~30s)

### Deploy to Render in 3 steps

1. **Fork this repo** to your GitHub account.

2. **Sign up** at <https://render.com> (free, no card needed).

3. **New Web Service**:
   - **Repository**: select this repo
   - **Build command**: `pip install -r requirements.txt && pip install -e .`
   - **Start command**: provided by `render.yaml` (auto-detected)
   - **Environment variables**:
     - `GROQ_API_KEY`: your key from <https://console.groq.com/keys>
     - `FLASK_SECRET_KEY`: a random string (Render can generate one)

The `render.yaml` blueprint is included in the repo and auto-configures 
everything (health check, threading, timeouts, Singapore region for speed).

**First deploy** takes ~3 min (builds the image, downloads the ONNX model). 
Later deploys cache the model and take ~30 sec. The **first API question** 
wakes the instance (~30s). Subsequent questions are instant.

## CI/CD pipeline

`.github/workflows/cicd.yaml` has three jobs:

1. **continuous-integration** (GitHub-hosted, every push & PR) -- installs
   dependencies (CPU-only PyTorch for speed) and runs the pytest suite. Uses
   no secrets, so it's safe on PRs from forks.
2. **build-and-push-ecr-image** (GitHub-hosted, `main` only) -- builds the
   Docker image and pushes it to ECR tagged `latest` and with the commit SHA.
3. **continuous-deployment** (**self-hosted runner on your EC2 instance**) --
   pulls the image, replaces the running container, and waits for `/health`.

## AWS deployment guide

One EC2 instance pulling images from a private ECR repository, fully
automated after this one-time setup.

> Security note: broad IAM policies are convenient for a demo. For anything
> longer-lived, scope them down to exactly the ECR/EC2 actions needed.

### 1. Create an IAM user

1. AWS Console -> **IAM** -> **Users** -> **Create user**, e.g. `autocode-analyzer-deployer` (no console access).
2. Attach `AmazonEC2ContainerRegistryFullAccess` and `AmazonEC2FullAccess`.
3. **Security credentials -> Create access key -> Command Line Interface (CLI)**.
   Save the **Access Key ID** and **Secret Access Key** (the secret is shown once).

### 2. Create an ECR repository

AWS Console -> **ECR** -> **Create repository** -> name it `autocode-analyzer` (private).

### 3. Launch an EC2 instance

1. **EC2** -> **Launch instance** -> AMI **Ubuntu Server 24.04 LTS**.
2. Instance type: **`t3.small` (2 GB RAM) recommended**. PyTorch + the
   embedding model + Chroma need roughly 1 GB, so a 1 GB `t2.micro`/`t3.micro`
   (free tier) only works with swap -- see below.
3. Storage: **20 GB** (the default 8 GB fills up after a few image pulls).
4. Security group inbound rules: SSH (22) from **My IP**, Custom TCP **8080** from `0.0.0.0/0`.
5. Launch and note the **public IPv4 address**.

### 4. Install Docker on the instance

```bash
ssh -i your-key.pem ubuntu@<public-ip>

sudo apt-get update -y
sudo apt-get install -y docker.io curl
sudo usermod -aG docker ubuntu
sudo systemctl enable docker --now
exit   # log out and back in so the docker group applies
```

Only if you chose a 1 GB free-tier instance, add 2 GB of swap:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 5. Register the instance as a self-hosted GitHub Actions runner

GitHub repo -> **Settings -> Actions -> Runners -> New self-hosted runner ->
Linux**. Run the commands GitHub shows (they contain a one-time token), e.g.:

```bash
mkdir actions-runner && cd actions-runner
# download + extract commands exactly as shown on the GitHub page, then:
./config.sh --url https://github.com/<you>/AI-END-TO-END-CODE-ANALYSIS --token <TOKEN>
sudo ./svc.sh install ubuntu
sudo ./svc.sh start
```

The service keeps the runner online across disconnects and reboots. Check it
shows **Idle** under Settings -> Actions -> Runners.

### 6. Add GitHub Secrets

Repo -> **Settings -> Secrets and variables -> Actions -> New repository secret**:

| Secret name             | Value                                             |
|-------------------------|---------------------------------------------------|
| `AWS_ACCESS_KEY_ID`     | from step 1                                        |
| `AWS_SECRET_ACCESS_KEY` | from step 1                                        |
| `AWS_REGION`            | e.g. `us-east-1` (where you created the ECR repo)  |
| `ECR_REPOSITORY_NAME`   | `autocode-analyzer`                                |
| `GROQ_API_KEY`          | your key from <https://console.groq.com/keys>      |
| `FLASK_SECRET_KEY`      | any long random string                             |

### 7. Deploy

Push to `main` (or **Actions -> CI/CD -> Run workflow**). When all three jobs
are green, open `http://<ec2-public-ip>:8080`.

### Cleanup (avoid AWS charges)

```bash
# on the EC2 instance
docker rm -f autocode-analyzer
cd ~/actions-runner && sudo ./svc.sh stop && sudo ./svc.sh uninstall
./config.sh remove --token <NEW_REMOVAL_TOKEN_FROM_GITHUB>
```

Then in the AWS Console: **terminate** the EC2 instance, delete the **ECR**
repository, and delete the IAM user's access key.

## Known limitations

- One analyzed repo at a time, held in memory: analyzing a new repo replaces
  the previous one, and restarting the server means clicking Analyze again.
  A multi-user version would key pipelines by session or repo.
- Only public repositories.
- Very large repos are capped at `MAX_FILES` (default 1500) files.
- Groq's free tier is rate-limited (about 30 requests/minute) -- fine for demos.

## Future improvements

- Per-session vector stores for multiple concurrent users.
- Streaming answers token by token.
- Private repos via a GitHub token.
- Re-use an existing index on restart instead of re-cloning.

## License

See [LICENSE](LICENSE).
