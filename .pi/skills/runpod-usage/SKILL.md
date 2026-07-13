---
name: runpod-usage
description: "Operate RunPod pods for dead-keys-census: query, resume, stop, migrate to replacement pods, discover SSH endpoints, configure persistent caches, run GPU smoke tests, stop stray workloads, and generate pods on demand in the same datacenter as a network volume."
---

# RunPod Usage

Use this skill when working with RunPod for this project, especially for GPU execution, pod lifecycle operations, SSH access, network-volume reuse, or CUDA smoke testing.

## Safety Rules

- MUST NOT print, commit, or paste secret values such as `RUNPOD_API_KEY` or `RUNPOD_SECRET_GITHUB_TOKEN`.
- MUST query live RunPod state before relying on pod runtime IPs, SSH ports, or desired status.
- MUST attach a network volume at pod creation/deployment time; do not assume it can be attached later to an arbitrary existing pod.
- MUST keep idle pods stopped unless the user explicitly asks for a running pod.
- For destructive or cost-incurring actions, get explicit user approval unless the current task already granted it.

## References

Project notes live in `AGENTS.md` at repository root. Treat them as authoritative for project-specific pod names, wrappers, environment variables, and historical discoveries.

Known non-secret project context as of 2026-07-10:

- Original pod: `dead-weight`, id `6mwc5q4jwwcgw9`.
- Migration pod: `dead-weight-migration`, id `lszgheen2t7qor`.
- Additional migration pod may exist from later work, e.g. `dead-weight-migration-2`; always query live state by name.
- Preferred idle state: `desiredStatus: EXITED`.
- Local RunPod credential env var: `RUNPOD_API_KEY`.
- In-pod GitHub credential env var: `RUNPOD_SECRET_GITHUB_TOKEN`.
- Use `scripts/cuda-run` / `scripts/cuda-python` on NVIDIA RunPod hosts.
- Reusable private RunPod template: `dead-keys-census-cuda` (id `1zpm2v05rn`); see Template-Based Pod Provisioning below.

## Query Pod State

TRIGGER: Before any lifecycle action, SSH connection, runtime-port use, or status report.

ACTION: Run a GraphQL query using the local `RUNPOD_API_KEY` and inspect only non-secret metadata.

```bash
curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
  -H 'content-type: application/json' \
  --data-binary '{"query":"query { myself { pods { id name desiredStatus imageName containerDiskInGb volumeInGb volumeMountPath ports runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } machine { gpuDisplayName cpuCount memoryTotal secureCloud machineType } } } }"}' \
  | jq '.data.myself.pods[] | select(.name|test("dead-weight"))'
```

OUTPUT: Pod id, name, desired status, runtime ports if running, and GPU/machine metadata.

ERROR RECOVERY:

- If `runtime` is `null`, the pod is not running or runtime metadata is unavailable; do not attempt runtime SSH until resumed.
- If the pod is missing, list all pods with `.data.myself.pods[] | {id,name,desiredStatus}` and ask/confirm the target.

## Resume a Pod

TRIGGER: User asks to start/resume an existing pod.

ACTION: Query state first, then call `podResume` for the target pod id.

```bash
curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
  -H 'content-type: application/json' \
  --data-binary '{"query":"mutation { podResume(input: {podId: \"lszgheen2t7qor\"}) { id name desiredStatus } }"}'
```

VERIFY:

```bash
curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
  -H 'content-type: application/json' \
  --data-binary '{"query":"query { myself { pods { id name desiredStatus runtime { ports { ip isIpPublic privatePort publicPort type } } machine { gpuDisplayName } } } }"}' \
  | jq '.data.myself.pods[] | select(.id=="lszgheen2t7qor")'
```

ERROR RECOVERY:

- If resume fails with insufficient free GPUs on the host, create/deploy a replacement pod against the same network volume in the same datacenter.
- Do not repeatedly retry a failed resume without checking capacity or replacement options.

## Stop a Pod

TRIGGER: User asks to stop/turn off a pod, or work is complete and the pod should be idle.

ACTION: Call `podStop` for the verified pod id.

```bash
curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
  -H 'content-type: application/json' \
  --data-binary '{"query":"mutation { podStop(input: {podId: \"lszgheen2t7qor\"}) { id name desiredStatus } }"}'
```

VERIFY: Query state and confirm `desiredStatus` is `EXITED`.

## SSH Discovery and Access

TRIGGER: Need shell access to a running pod.

ACTION:

1. Query `runtime.ports` live.
2. Prefer public TCP SSH endpoint when available.
3. If public TCP SSH fails or is unavailable, use the RunPod web UI Connect/SSH command for the full proxy username.

Public TCP shape:

```bash
ssh -p <publicPort> root@<public-ip> -i ~/.ssh/id_ed25519
```

Proxy shape:

```bash
ssh <pod-id>-<suffix>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

For scripted proxy commands, allocate a PTY:

```bash
printf '%s\n' 'cd /workspace/dead-keys-census && nvidia-smi' \
  | ssh -tt <pod-id>-<suffix>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

NOTES:

- The pod id alone is not enough for the `ssh.runpod.io` username; the suffix comes from the RunPod UI Connect command.
- Runtime IPs and ports can change after restart; do not reuse stale values without querying.

## Persistent Cache Setup on Network Volume

TRIGGER: A replacement pod has been created, dependencies need installation, or models may be downloaded.

ACTION: Run the project setup script inside the pod before heavyweight installs/downloads.

```bash
cd /workspace/dead-keys-census
./scripts/runpod-persistent-cache-setup
. ~/.dead-keys-census-runpod-env
```

If auto-detection fails:

```bash
DEAD_KEYS_PERSISTENT_CACHE_ROOT=/workspace/dead-keys-census-cache ./scripts/runpod-persistent-cache-setup
. ~/.dead-keys-census-runpod-env
```

VERIFY:

```bash
env | grep -E '^(HF_HOME|HUGGINGFACE_HUB_CACHE|TRANSFORMERS_CACHE|TORCH_HOME|PIP_CACHE_DIR|UV_CACHE_DIR|TRITON_CACHE_DIR|DEAD_KEYS_CUDA_VENV)='
test -d "$DEAD_KEYS_CUDA_VENV" || echo "CUDA venv will be created on first cuda-run"
```

## CUDA Environment and Smoke Tests

TRIGGER: Need to verify that Phase 1 / Phase 1.5 workloads will use the NVIDIA GPU.

ACTION: Use project CUDA wrappers, not the local ROCm `uv` environment.

```bash
./scripts/cuda-run - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.ones(16, device='cuda') + 2
torch.cuda.synchronize()
print(x.device, x[:3].cpu().tolist())
PY
```

Phase 1 smoke test:

```bash
./scripts/cuda-run -m phase1.scripts.census --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024
```

For Pythia smoke tests, use the project scripts/instructions current in the repository and keep limits small. Do not start full runs unless explicitly requested.

VERIFY:

```bash
nvidia-smi
find outputs -maxdepth 1 -type f | sort
```

## Stop Stray Workloads

TRIGGER: A workload is running unexpectedly, using CPU instead of GPU, or user asks to stop active work in a pod.

ACTION:

1. Inspect active processes and GPU usage.
2. Identify project workload PIDs.
3. Stop only the confirmed workload processes.
4. Verify CPU/GPU utilization falls and no target workload remains.

Commands:

```bash
ps -eo pid,ppid,stat,pcpu,pmem,etime,cmd --sort=-pcpu | head -40
nvidia-smi
pgrep -af 'phase1|census|pythia|python'
```

Stop pattern after confirming PIDs:

```bash
kill <pid>
sleep 5
ps -p <pid> -o pid,stat,cmd || true
```

If needed:

```bash
kill -TERM <pid>
sleep 10
kill -KILL <pid>
```

## Replacement Pod / Migration with Existing Network Volume

TRIGGER: Existing pod cannot resume due to host capacity, or a fresh pod is needed with the same persistent workspace/cache.

ACTION:

1. Identify the network volume and its datacenter.
2. Choose a pod configuration available in that same datacenter.
3. Create/deploy the pod with the network volume attached at creation time.
4. Query and record pod id, name, GPU, image, runtime SSH ports, and mounted volume path.
5. Run persistent cache setup before installs/downloads.
6. Smoke test CUDA.
7. Stop the pod when idle.

CRITICAL: For cross-datacenter moves, use two running pods and `rsync` between `/workspace` mounts. Do not assume a volume can be attached across datacenters.

## Generate Pod On Demand in Same Datacenter as Volume

TRIGGER: User asks to create/generate a RunPod pod from a requested GPU, CPU, memory configuration and an existing network volume.

INPUTS:

- GPU requirement, e.g. `RTX A4500`, `RTX 4090`, or minimum VRAM/GPU count.
- CPU requirement, e.g. minimum vCPU count.
- Memory requirement, e.g. minimum system RAM in GB.
- Network volume id or name.
- Optional pod name, image/template, container disk size, ports, and environment variables.

ACTION:

1. Query network volumes and determine the target volume datacenter.
2. Query available GPU/machine offerings in that datacenter.
3. Filter offerings by GPU, CPU, and memory requirements.
4. Select the cheapest/most available compatible secure-cloud offering unless the user specifies otherwise.
5. Create/deploy the pod with the selected offering and attach the network volume at creation time.
6. Verify the pod exists, is in the same datacenter as the volume, and mounts the expected volume path.
7. Run SSH discovery, persistent cache setup, and CUDA smoke test.
8. Report pod id/name, GPU, CPU, memory, datacenter, volume, SSH access method, and verification evidence.

DISCOVERY TEMPLATE:

RunPod GraphQL schema names change over time. Use this as the required shape, but introspect or check current docs for exact fields before mutation.

```bash
# 1. Find the target network volume and datacenter.
curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
  -H 'content-type: application/json' \
  --data-binary '{"query":"query { myself { networkVolumes { id name dataCenterId size } } }"}' \
  | jq '.data.myself.networkVolumes[]'

# 2. Find compatible machines/GPU types in that datacenter.
curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
  -H 'content-type: application/json' \
  --data-binary '{"query":"query { gpuTypes { id displayName memoryInGb secureCloud communityCloud lowestPrice(input: {gpuCount: 1}) } }"}' \
  | jq '.'
```

SELECTION RULE:

```text
candidate.datacenter == network_volume.datacenter
AND candidate.gpu satisfies requested GPU/VRAM/count
AND candidate.cpuCount >= requested CPU
AND candidate.memoryTotalGb >= requested memory
```

CREATE TEMPLATE:

Before running this mutation, verify exact input fields against current RunPod docs/schema. The important invariant is `networkVolumeId` plus same datacenter selection.

```bash
curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
  -H 'content-type: application/json' \
  --data-binary @- <<'JSON'
{
  "query": "mutation DeployPod($input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: $input) { id name desiredStatus machine { gpuDisplayName cpuCount memoryTotal } } }",
  "variables": {
    "input": {
      "name": "dead-weight-on-demand",
      "cloudType": "SECURE",
      "gpuTypeId": "<gpu-type-id>",
      "gpuCount": 1,
      "minVcpuCount": <cpu-count>,
      "minMemoryInGb": <memory-gb>,
      "dataCenterId": "<volume-datacenter-id>",
      "networkVolumeId": "<network-volume-id>",
      "imageName": "<image>",
      "containerDiskInGb": 50,
      "ports": "22/tcp"
    }
  }
}
JSON
```

VERIFY AFTER CREATE:

```bash
curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
  -H 'content-type: application/json' \
  --data-binary '{"query":"query { myself { pods { id name desiredStatus runtime { ports { ip isIpPublic privatePort publicPort type } } machine { gpuDisplayName cpuCount memoryTotal secureCloud machineType } } } }"}' \
  | jq '.data.myself.pods[] | select(.name=="dead-weight-on-demand")'
```

Then SSH in and verify:

```bash
mount | grep -E '/workspace|runpod'
nvidia-smi
cd /workspace/dead-keys-census && ./scripts/runpod-persistent-cache-setup
cd /workspace/dead-keys-census && ./scripts/cuda-run - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

ERROR RECOVERY:

- If no compatible offering exists in the volume datacenter, STOP and report the unsatisfied constraints; do not create a pod in another datacenter.
- If the schema rejects the mutation, introspect/check current RunPod docs and adjust field names while preserving the same-datacenter and network-volume-at-creation invariants.
- If pod starts without the volume mounted, stop investigation before installing dependencies; report the mismatch and do not download models to ephemeral disk.

## Template-Based Pod Provisioning

TRIGGER: A replacement pod is needed and the project's reusable RunPod template is the preferred provisioning path (faster and less error-prone than re-specifying image, disk, ports, and mount path each time).

CONTEXT:

- Private template name: `dead-keys-census-cuda`
- Template id: `1zpm2v05rn`
- Verified 2026-07-12 by deploying `dead-keys-template-smoke-delete-me` from the template on an L4, then stopping/deleting the smoke pod.
- The template captures reusable pod configuration only: CUDA PyTorch image, 30 GB container disk, NVIDIA category, `/workspace` mount path, and `22/tcp` SSH port. It does NOT snapshot network-volume contents or Git checkout state.
- Keep heavyweight model/package caches and generated results on the attached network volume; run `./scripts/runpod-persistent-cache-setup` after first SSH into a pod deployed from the template.

ACTION — Create or recreate the template via REST API:

```bash
cat > /tmp/deadkeys_template.json <<'JSON'
{
  "category": "NVIDIA",
  "containerDiskInGb": 30,
  "dockerEntrypoint": [],
  "dockerStartCmd": [],
  "env": {},
  "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  "isPublic": false,
  "isServerless": false,
  "name": "dead-keys-census-cuda",
  "ports": ["22/tcp"],
  "readme": "Dead Keys Census CUDA pod template: base PyTorch CUDA image, SSH TCP, /workspace network-volume mount expected at deploy time. Keep model/cache/venv data on the network volume, not in the template.",
  "volumeInGb": 0,
  "volumeMountPath": "/workspace"
}
JSON

curl -sS --request POST \
  --url https://rest.runpod.io/v1/templates \
  --header "Authorization: Bearer $RUNPOD_API_KEY" \
  --header 'Content-Type: application/json' \
  --data @/tmp/deadkeys_template.json \
  | jq '{id,name,imageName,category,containerDiskInGb,volumeInGb,volumeMountPath,ports}'
```

ACTION — Deploy a replacement pod from the template:

```bash
cat > /tmp/deadkeys_template_pod.json <<'JSON'
{
  "name": "dead-weight-migration-N",
  "templateId": "1zpm2v05rn",
  "gpuTypeIds": ["NVIDIA L4"],
  "gpuCount": 1,
  "cloudType": "SECURE"
}
JSON

curl -sS --request POST \
  --url https://rest.runpod.io/v1/pods \
  --header "Authorization: Bearer $RUNPOD_API_KEY" \
  --header 'Content-Type: application/json' \
  --data @/tmp/deadkeys_template_pod.json \
  | jq '{id,name,desiredStatus,imageName,templateId,gpuCount,machineId}'
```

VERIFY: Query pod state (see Query Pod State) and confirm the new pod is running with the expected image and an attached `/workspace` volume; then run SSH discovery, persistent cache setup, and a CUDA smoke test as in the Replacement Pod / Migration section.

ACTION — Stop and delete a smoke or throwaway pod immediately when no longer needed:

```bash
curl -sS --request POST \
  --url "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
  --header "Authorization: Bearer $RUNPOD_API_KEY" \
  | jq '{id,name,desiredStatus}'

curl -sS --request DELETE \
  --url https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID \
  --header "Authorization: Bearer $RUNPOD_API_KEY"
```

ERROR RECOVERY:

- If a pod deployed from the template comes up without the network volume mounted, STOP before installing dependencies or downloading models; the template only sets the mount path, the volume itself must be available and attachable at deploy time in the chosen datacenter.
- If the REST endpoint or field names are rejected, introspect the current RunPod REST docs and adjust while preserving the `templateId` + `cloudType: SECURE` + GPU type invariants.
- Do not leave smoke pods running; stop and delete them in the same session that created them.

## Reporting Template

```text
Status: DONE | NEEDS_INPUT | BLOCKED | FAILED

Summary:
- ...

Actions:
- ...

Verification:
- command: ...
  evidence: ...

Open Issues:
- ...
```
