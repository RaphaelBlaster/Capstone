# Complete private archive after final experiment

User explicitly authorized all outputs, weights and the original image datasets on09October2026. This supersedes the earlier exclusion of weights/datasets from GitHub for this archive. Ordinary Git history still excludes large binaries and credentials.

One terminal-only CPU backup worker follows the existing final Kaggle Version1. It never changes submitted code, launches/retries GPU work, or submits a competition prediction. The saved Kaggle execution continues independently of a browser/chat. Completed saved outputs persist remotely; abrupt termination may lose recent working-state writes.

Destination: private RaphaelBlaster/Img2BioMass release jepa-block-shuffle-v1-137902867. All Kaggle working outputs, exact submitted notebook, adapted.pt and its provenance, CSIRO datasetversion2, GrassClover datasetversion1 and official I-JEPA input datasetversion1. Original dataset snapshots are explicitly pinned; their relationship to historical inputs is documented by saved manifests/hashes, not assumed from a slug.

Streaming1GiB parts avoids normal Git's100MB limit and GitHub release's2GiB-per-asset limit, and bounds staging disk while copying the full GrassClover corpus. Each part and reconstructed stream has byte counts and SHA256. Verify GitHub's server digest when returned; otherwise only server byte length and local digest are verified. All file bytes inside the experiment tar have their own hashes. Existing source checkpoints remain local and on Kaggle. Only new staging parts are removed after successful upload readback.

The release remains draft until every stream, manifest and restoration document is uploaded. A network/authentication or ambiguous write stops and preserves journals; no clobber or automatic mutation retry. Publication occurs only after a fresh private-repository check. Current worker and pending/uploaded status are under outputs/block_shuffle_github_archive/. A queued worker is not evidence that any archive has already been uploaded.

The exact implementation is the submitted notebook's embedded code. The release tag can point at the earlier repository commit; do not claim new code is committed merely because its immutable notebook is archived. Dataset/Meta attribution and licenses remain applicable. No secrets, login files or private conversation exports are added.
