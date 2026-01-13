# Advanced Usage

This guide covers advanced topics such as batch processing, cluster submission, and hardware configuration.

## Batch Processing

`qmri-neuropipe` provides built-in support for processing multiple subjects in parallel using the `-j / --jobs` flag.

### Basic Parallel Execution
To process multiple subjects locally:
```bash
qmri-neuropipe --config config.yaml -p sub-01 -p sub-02 --jobs 4
```
This will run 4 subjects concurrently.

### Using a Subjects File
For large batches, use a subjects file instead of listing IDs on the command line:

```bash
qmri-neuropipe --config config.yaml --subjects-file subjects.txt --jobs 8
```

The `subjects.txt` file format is simple:
```
sub-001, ses-01
sub-002, ses-01
sub-003
```
(Session is optional)

## HTCondor Submission

To generate submit files for an HTCondor cluster instead of running locally, use the `--submit` flag.

```bash
qmri-neuropipe --config config.yaml --subjects-file subjects.txt --submit
```

This will create `submit_qmri_sub-XXX.sub` files in your output directory (or working directory). You can then submit them using `condor_submit`.

## Hardware Configuration

### CPU Configuration
- `--n-cpus`: Controls how many threads *each pipeline instance* uses (e.g., for ANTs, MRtrix).
- `--omp-nthreads`: Explicitly sets `OMP_NUM_THREADS`. If not set, it defaults to `n_cpus`.

**Recommendation for Clusters:**
Set `--n-cpus` to the number of Cores requested per job.

### GPU Configuration
To enable GPU acceleration (e.g., for `eddy_cuda`):

```bash
qmri-neuropipe ... --use-gpu
```

#### Multi-GPU processing
If you have multiple GPUs on a single node (or local machine) and want to distribute jobs across them:

```bash
qmri-neuropipe ... --jobs 4 --gpu-ids "0,1,2,3"
```
The pipeline will automatically assign each parallel worker a specific GPU ID in a round-robin fashion.
- Worker 1 -> GPU 0
- Worker 2 -> GPU 1
- ...
- Worker 5 -> GPU 0

This ensures optimal utilization of multi-GPU resources.
