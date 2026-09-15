"""
clipping.cloud — PC-prep + Kaggle-render split, transported via a Kaggle
Dataset "mailbox" instead of a live cloud store (no GCS/Drive account
needed). See:

  job_store.py         PC-side durable queue state (READY/SENT/COMPLETED/FAILED)
  manifest.py          portable job manifest <-> prepare_pipeline() dict
  kaggle_transport.py  batched Kaggle Dataset push/pull
  pc_prep_loop.py       PC entry point: prepare videos, push batches to Kaggle
  kaggle_consumer.py    Kaggle notebook entry point: render, push results back
"""
