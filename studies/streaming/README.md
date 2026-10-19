# Memory measured on a complete large tensor

`benchmark.py` makes labelled synthetic base/merged FP32 matrices of shape
4096 x 8192, each 128 MiB of tensor data, and rank-8 LoRA factors. Every one of
33,554,432 entries has an exactly representable update of 1/128. A complete
binary64 delta alone would occupy 256 MiB. The auditor instead calculates and
discards row blocks, checks every entry and hashes every input file.

The [measurement receipt](results/benchmark.json) records process peak RSS and
elapsed audit time at 1, 16 and 64 MiB block settings. Measurements use a fresh
worker process per setting on WSL2 Linux with 14 logical CPUs and one OpenBLAS/OMP
thread. Elapsed time includes input validation, arithmetic and artifact hashes;
it excludes interpreter/import startup and synthetic input creation. This is
a warm-cache local measurement, not a cross-machine speed claim.
The retained installed-wheel run used the shared host under ordinary session
activity; its elapsed times are not an isolated throughput benchmark. Process RSS is the
quantity used for the README's memory result.

Preparation has its own short-lived child process. Preparing a dense input in
the parent would contaminate the Linux peak-RSS value inherited by later workers.
The process RSS includes Python, NumPy, safetensors, A and BLAS, not just the
working blocks. The block option is a target rather than a hard RSS cap.

All settings must report every entry changed, zero residual and identical
concrete evidence. The temporary large input weights are not committed.

```sh
OPENBLAS_NUM_THREADS=1 uv run python studies/streaming/benchmark.py /tmp/mergeglass-streaming
```

Use a new output directory. Linux `resource.ru_maxrss` is measured in KiB.
Other platforms can use different units; the retained receipt is a Linux result.

## The actual public checkpoint

The [public-case receipt](results/public-memory.json) measures the installed
Python 3.15 CLI on the unchanged public SmolLM2 adapter and its native BF16 merge.
It checks all 272 tensors and fingerprints 540,004,492 input-file bytes, with
62,708 KiB (61.24 MiB) peak process RSS. The input hashes and exact runtime module
hashes are retained; PyTorch is absent from this lightweight environment.

The command runs in the prepared public-study directory:

```sh
OPENBLAS_NUM_THREADS=1 /usr/bin/time -f '{"peak_rss_kib":%M}' -o /tmp/public-memory.json mergeglass base adapter merged-bf16 --summary
```

GNU time measures a fresh CLI process; no parent prepares or retains a dense
matrix. The output is byte-identical to the retained public terminal capture.
The summary mode performs the complete audit and produces the same full JSON
as the per-layer view. This single shared-host measurement describes process
RSS, not filesystem cache usage, a hard memory cap, or isolated throughput.
