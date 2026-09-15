"""
clipping.cloud.kaggle_transport — batched Kaggle Dataset push/pull.

Used as the transport layer between the PC prep loop and the Kaggle GPU
consumer. Kaggle Dataset versioning is a FULL SNAPSHOT model, not an
incremental queue: each push replaces the dataset's file list with
whatever's currently in the local folder. Two consequences callers must
respect:

  1. The PC must prune manifests for videos it has already reconciled as
     COMPLETED from its local inbox folder before the next push, or
     they'll keep reappearing forever.
  2. The Kaggle consumer must track which video_ids it has already
     submitted to its render farm this session, since a re-pull can
     legitimately contain manifests it already has (nothing wrong with
     that — it's just a resend of the same snapshot plus whatever's new).

Real job state lives in clipping.cloud.job_store (PC side) and the
consumer's own in-memory "already submitted" set (Kaggle side) — this
module only moves bytes.
"""

import os

import kaggle


def push_dataset_version(local_dir: str, message: str, quiet: bool = False) -> None:
    """
    Push the current contents of local_dir as a new version of the Kaggle
    Dataset described by local_dir/dataset-metadata.json (created via
    `kaggle datasets init` / `kaggle datasets create` once, up front).
    """
    kaggle.api.authenticate()
    kaggle.api.dataset_create_version(
        folder=local_dir,
        version_notes=message,
        dir_mode="zip",
        quiet=quiet,
    )


def pull_dataset(dataset_ref: str, dest_dir: str, quiet: bool = False) -> None:
    """Download + unzip the latest version of a Kaggle Dataset into dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_ref, path=dest_dir, unzip=True, quiet=quiet)
