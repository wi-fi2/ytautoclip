"""
web.api.routes.cloud — Cloud Batch page endpoints.

Controls the PC-side prep loop (clipping.cloud.pc_prep_loop) that feeds a
Kaggle dual-GPU render session via a Kaggle Dataset mailbox — see
clipping/cloud/ for the full design. This API only supervises the PC half;
starting the actual Kaggle notebook session is a manual step done in the
Kaggle UI once the dashboard shows "batch ready".
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import cloud_manager
from ..models import CloudBatchBacklogRequest, CloudBatchStartRequest, CloudBatchStatusResponse

router = APIRouter(prefix="/api/cloud", tags=["cloud"])


@router.get("/status")
async def get_status() -> CloudBatchStatusResponse:
    return CloudBatchStatusResponse(**cloud_manager.status())


@router.get("/backlog")
async def get_backlog() -> dict:
    return {"urls": cloud_manager.get_backlog()}


@router.post("/backlog")
async def add_backlog(req: CloudBatchBacklogRequest) -> dict:
    total = cloud_manager.add_urls(req.urls)
    return {"backlog_count": total}


@router.delete("/backlog")
async def clear_backlog() -> dict:
    cloud_manager.clear_backlog()
    return {"backlog_count": 0}


@router.post("/start")
async def start_prep_loop(req: CloudBatchStartRequest) -> CloudBatchStatusResponse:
    try:
        cloud_manager.start(req.inbox_dataset, req.outbox_dataset, req.threshold_clips, req.prep_workers)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CloudBatchStatusResponse(**cloud_manager.status())


@router.post("/stop")
async def stop_prep_loop() -> CloudBatchStatusResponse:
    cloud_manager.stop()
    return CloudBatchStatusResponse(**cloud_manager.status())
