"""Run a natural cell on Modal — free, serverless, unattended (no tab, no sleep, no 12h wall).

Modal's free tier ($30/month credits, ~187 T4-hours) runs this headless in the cloud. Setup once:

    pip install modal
    modal token new                # browser auth, one time

Then fire a run (use --detach so it keeps running after you close the terminal):

    modal run --detach scripts/modal_run.py --dataset 2wiki \\
        --model mistralai/Mistral-7B-Instruct-v0.3 --tag mistral --mscs-limit 80

Results land in the persistent 'dragnet-runs' volume; pull them locally with:

    modal volume get dragnet-runs /2wiki ./runs

Models cache in a second volume so only the first run downloads weights. Repos are baked at
image-build time — after pushing repo changes, add --force-build once to refresh the image.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "git clone --depth 1 https://github.com/santoshcheethiralame-dot/LINEUP /root/lineup",
        # the code repo is still named SCoPE on GitHub (DRAGNET rename pending); this URL works now
        # and redirects to DRAGNET automatically once renamed, so it is correct in both states.
        "git clone --depth 1 https://github.com/santoshcheethiralame-dot/SCoPE /root/dragnet",
        "pip install -e '/root/lineup[gpu]'",
        "pip install --no-deps -e /root/dragnet",
        "pip install -U 'bitsandbytes>=0.46.1'",
    )
)

app = modal.App("dragnet")
runs_vol = modal.Volume.from_name("dragnet-runs", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=8 * 60 * 60,
    volumes={"/root/runs": runs_vol, "/root/.cache/huggingface": hf_vol},
)
def natural_cell(dataset: str, model: str, tag: str, max_size: int, mscs_limit: int):
    import subprocess

    subprocess.run(
        [
            "python", "scripts/run_natural_cell.py",
            "--dataset", dataset, "--model", model, "--tag", tag,
            "--max-size", str(max_size), "--mscs-limit", str(mscs_limit),
            "--lineup-scripts", "/root/lineup/scripts", "--out-root", "/root/runs",
        ],
        cwd="/root/dragnet",
        check=True,
    )
    runs_vol.commit()   # persist the written cell so `modal volume get` can retrieve it


@app.local_entrypoint()
def main(
    dataset: str = "hotpotqa",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    tag: str = "qwen",
    max_size: int = 5,
    mscs_limit: int = 100,
):
    # spawn (not remote): the job runs server-side to completion even after the terminal exits or
    # the laptop closes. The entrypoint returns immediately with a handle.
    call = natural_cell.spawn(dataset, model, tag, max_size, mscs_limit)
    print(f"spawned {dataset}/natural/{tag}  (call {call.object_id}) — running on Modal; safe to close the laptop.")
