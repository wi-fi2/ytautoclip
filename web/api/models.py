"""
web.api.models — Pydantic schemas for request/response validation.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourcePlatform(str, enum.Enum):
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    GDRIVE = "gdrive"


class AspectRatio(str, enum.Enum):
    RATIO_9_16 = "9:16"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_3_4 = "3:4"
    RATIO_4_5 = "4:5"


class FontStyle(str, enum.Enum):
    DEFAULT = "DEFAULT"
    STORYTELLER = "STORYTELLER"
    HORMOZI = "HORMOZI"
    CINEMATIC = "CINEMATIC"


class FaceDetector(str, enum.Enum):
    MEDIAPIPE = "mediapipe"
    YOLO = "yolo"


class AIProvider(str, enum.Enum):
    GEMINI = "gemini"
    NVIDIA = "nvidia"


class WhisperDevice(str, enum.Enum):
    CUDA = "cuda"
    CPU = "cpu"
    AUTO = "auto"


# ---------------------------------------------------------------------------
# Job Creation Request
# ---------------------------------------------------------------------------

class JobMode(str, enum.Enum):
    AUTO = "auto"                    # AI picks the clips (default, existing behavior)
    DOWNLOAD_ONLY = "download_only"  # Download (+ nothing else), used to stage manual selection
    MANUAL = "manual"                # Render exactly the given manual_segments, no AI analysis


class ManualSegment(BaseModel):
    """A user-picked [start, end] window to render as its own clip."""
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., gt=0)
    title: str = Field("", description="Optional label shown as the clip title")


class JobCreateRequest(BaseModel):
    """Payload to create a new clipping job."""

    # Source
    url: Optional[str] = Field(None, description="Video URL to process")
    upload_filename: Optional[str] = Field(None, description="Filename of an uploaded video")
    source: SourcePlatform = Field(SourcePlatform.YOUTUBE, description="Source platform")
    reuse_job_id: Optional[str] = Field(None, description="Existing Job ID to reuse its downloads and JSON")

    # Clip selection mode
    mode: JobMode = Field(JobMode.AUTO, description="auto = AI picks clips; download_only = fetch video then stop (for manual picking); manual = render manual_segments as-is")
    manual_segments: list[ManualSegment] = Field(default_factory=list, description="Used when mode == 'manual'")

    # Main settings
    clips: int = Field(7, ge=1, le=30, description="Number of clips to generate")
    ratio: AspectRatio = Field(AspectRatio.RATIO_9_16, description="Output aspect ratio")
    source_height: str = Field("max", description="Source download max height")
    render_height: str = Field("1080", description="Target output height")

    # Content & Hook
    words_per_sub: int = Field(5, ge=1, le=15)
    hook_duration: int = Field(3, ge=1, le=10)
    use_broll: bool = True
    use_hook_glitch: bool = True
    use_auto_bgm: bool = True
    use_karaoke_effect: bool = True
    use_split_screen: bool = False
    use_camera_switch: bool = False
    no_subs: bool = False

    # Hook V2
    hook_v2: bool = False
    hook_v2_items: int = Field(3, ge=2, le=6)
    no_segment_trim: bool = False
    silence_trim: bool = False

    # Subtitle & Typography
    font_style: FontStyle = FontStyle.HORMOZI

    # Whisper
    whisper_model: str = "large-v3"
    whisper_device: WhisperDevice = WhisperDevice.AUTO
    whisper_compute_type: str = "float16"
    use_dlp_subs: bool = False

    # AI
    ai_provider: AIProvider = AIProvider.GEMINI
    gemini_model: str = "gemini-3-flash-preview"
    face_detector: FaceDetector = FaceDetector.MEDIAPIPE

    # GPU rendering — spreads this job's own clips across however many
    # physical GPUs are visible (e.g. dual T4), see clipping.render_farm.
    render_gpus: str = Field("auto", description="'auto' uses every visible GPU, or an integer to cap it")
    render_workers_per_gpu: int = Field(1, ge=1, le=4, description="Render processes per physical GPU — raise only after benchmarking clips/hour")


# ---------------------------------------------------------------------------
# Job Progress Event
# ---------------------------------------------------------------------------

class JobProgressEvent(BaseModel):
    """Single progress event for SSE streaming."""
    step: str
    step_number: int
    total_steps: int
    message: str
    percent: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Clip Detail
# ---------------------------------------------------------------------------

class ClipDetail(BaseModel):
    """Metadata for a single rendered clip."""
    rank: int
    viral_score: Optional[int] = None
    title: Optional[str] = None
    title_en: Optional[str] = None
    filename: str
    duration: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    download_url: str
    thumbnail_url: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Job Response
# ---------------------------------------------------------------------------

class JobResponse(BaseModel):
    """Full job detail for API responses."""
    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    url: Optional[str] = None
    upload_filename: Optional[str] = None
    source: SourcePlatform = SourcePlatform.YOUTUBE
    config: dict = Field(default_factory=dict)
    progress: Optional[JobProgressEvent] = None
    clips: list[ClipDetail] = Field(default_factory=list)
    error: Optional[str] = None
    log: list[str] = Field(default_factory=list)
    source_video_url: Optional[str] = Field(
        None, description="URL to preview the downloaded source video, once available"
    )
    source_duration: Optional[float] = Field(
        None, description="Duration (seconds) of the downloaded source video, once known"
    )


class JobListResponse(BaseModel):
    """Paginated job list."""
    jobs: list[JobResponse]
    total: int


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class SettingsRequest(BaseModel):
    """Settings update payload."""
    google_api_key: Optional[str] = None
    pexels_api_key: Optional[str] = None
    hf_token: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    # Defaults
    default_clips: Optional[int] = None
    default_ratio: Optional[AspectRatio] = None
    default_font_style: Optional[FontStyle] = None
    default_whisper_model: Optional[str] = None
    default_whisper_device: Optional[WhisperDevice] = None
    default_ai_provider: Optional[AIProvider] = None


class SettingsResponse(BaseModel):
    """Current settings (keys are masked)."""
    google_api_key_set: bool = False
    pexels_api_key_set: bool = False
    hf_token_set: bool = False
    nvidia_api_key_set: bool = False
    default_clips: int = 7
    default_ratio: str = "9:16"
    default_font_style: str = "HORMOZI"
    default_whisper_model: str = "large-v3"
    default_whisper_device: str = "auto"
    default_ai_provider: str = "gemini"
    gpu_available: bool = False
    gpu_count: int = 0


class CloudBatchStartRequest(BaseModel):
    """Start the PC-side prep loop that feeds a Kaggle dual-GPU render session."""
    inbox_dataset: str = Field(..., description="e.g. yourname/ytclipper-inbox")
    outbox_dataset: str = Field(..., description="e.g. yourname/ytclipper-outbox")
    threshold_clips: int = Field(60, ge=1, description="Push a batch once this many clips are queued")
    prep_workers: int = Field(2, ge=1, le=8, description="Videos to prepare concurrently")


class CloudBatchBacklogRequest(BaseModel):
    """Add source URLs to the cloud batch backlog."""
    urls: list[str] = Field(..., min_length=1)


class CloudBatchStatusResponse(BaseModel):
    """Current state of the PC prep loop + job queue, for the Cloud Batch page."""
    running: bool = False
    pid: Optional[int] = None
    started_at: Optional[datetime] = None
    inbox_dataset: Optional[str] = None
    outbox_dataset: Optional[str] = None
    threshold_clips: Optional[int] = None
    backlog_count: int = 0
    job_counts: dict = Field(default_factory=dict)
    recent_log: list[str] = Field(default_factory=list)


class SystemHealthResponse(BaseModel):
    """System health check."""
    status: str = "ok"
    version: str
    gpu_available: bool
    ffmpeg_available: bool
    jobs_running: int
    jobs_queued: int
