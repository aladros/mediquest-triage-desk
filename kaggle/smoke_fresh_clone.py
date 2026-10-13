# Fresh-clone smoke test of the README's one command, on a Kaggle GPU (what a judge would do).
# Clones the PUBLIC repo, installs only the README prerequisites (uv + Ollama), then runs ./run.sh with a scripted nurse
# answering "c" (confirm) at every gate. Push with: uvx kaggle kernels push -p kaggle/build-smoke
import subprocess

def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

sh("(apt-get -qq update && apt-get -qq install -y pciutils lshw zstd) >/dev/null 2>&1 || true")
sh("curl -fsSL https://ollama.com/install.sh | sh >/dev/null 2>&1 && ollama --version")
sh("curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1")
sh("rm -rf /tmp/fresh && git clone -q https://github.com/aladros/mediquest-triage-desk /tmp/fresh && git -C /tmp/fresh log --oneline -1")
env = "export PATH=$HOME/.local/bin:$PATH; cd /tmp/fresh"
sh(f"{env}; uv run pytest -q")
sh(f"{env}; printf 'c\\nc\\nc\\nc\\nc\\nc\\n' | ./run.sh demo --nurse 'Smoke test (not a clinician)'")
sh(f"{env}; printf 'c\\n' | ./run.sh recheck --nurse 'Smoke test (not a clinician)'")
sh(f"{env}; ./run.sh queue")
sh(f"{env}; wc -l var/audit.jsonl")
print("SMOKE OK", flush=True)
