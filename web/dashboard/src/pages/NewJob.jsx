import { useState, useRef, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { createJob, uploadVideo, discoverVideos } from '../api'

function NewJob() {
  const navigate = useNavigate()
  const location = useLocation()
  const fileRef = useRef(null)

  const [mode, setMode] = useState('url') // 'url', 'upload', or 'reuse'
  const [clipMode, setClipMode] = useState('auto') // 'auto' (AI) or 'manual' (pick timestamps)
  const [url, setUrl] = useState('')
  const [uploadFilename, setUploadFilename] = useState('')
  const [uploading, setUploading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [reuseJobId, setReuseJobId] = useState('')

  // Auto-discovery
  const [discoverQuery, setDiscoverQuery] = useState('')
  const [discovering, setDiscovering] = useState(false)
  const [discoverError, setDiscoverError] = useState('')
  const [discoverResults, setDiscoverResults] = useState(null)

  // Config
  const [clips, setClips] = useState(7)
  const [ratio, setRatio] = useState('9:16')
  const [source, setSource] = useState('youtube')
  const [fontStyle, setFontStyle] = useState('HORMOZI')
  const [whisperModel, setWhisperModel] = useState('large-v3')
  const [whisperDevice, setWhisperDevice] = useState('auto')
  const [aiProvider, setAiProvider] = useState('gemini')

  // Toggles
  const [useBroll, setUseBroll] = useState(true)
  const [useHookGlitch, setUseHookGlitch] = useState(true)
  const [useBgm, setUseBgm] = useState(true)
  const [useKaraoke, setUseKaraoke] = useState(true)
  const [noSubs, setNoSubs] = useState(false)
  const [hookV2, setHookV2] = useState(false)
  const [silenceTrim, setSilenceTrim] = useState(false)
  const [useDlpSubs, setUseDlpSubs] = useState(false)
  const [loadGeminiJson, setLoadGeminiJson] = useState(false)

  // GPU rendering
  const [renderGpus, setRenderGpus] = useState('auto')
  const [renderWorkersPerGpu, setRenderWorkersPerGpu] = useState(1)

  // Load from location state if user clicked "Clone / Rerun"
  useEffect(() => {
    const reuseJob = location.state?.reuseJob
    if (reuseJob) {
      setReuseJobId(reuseJob.id)
      setUrl(reuseJob.url || '')
      setUploadFilename(reuseJob.upload_filename || '')
      setMode('reuse')
      setSource(reuseJob.source || 'youtube')
      
      const config = reuseJob.config || {}
      if (config.clips !== undefined) setClips(config.clips)
      if (config.ratio !== undefined) setRatio(config.ratio)
      if (config.font_style !== undefined) setFontStyle(config.font_style)
      if (config.whisper_model !== undefined) setWhisperModel(config.whisper_model)
      if (config.whisper_device !== undefined) setWhisperDevice(config.whisper_device)
      if (config.ai_provider !== undefined) setAiProvider(config.ai_provider)
      
      if (config.use_broll !== undefined) setUseBroll(config.use_broll)
      if (config.use_hook_glitch !== undefined) setUseHookGlitch(config.use_hook_glitch)
      if (config.use_auto_bgm !== undefined) setUseBgm(config.use_auto_bgm)
      if (config.use_karaoke_effect !== undefined) setUseKaraoke(config.use_karaoke_effect)
      if (config.hook_v2 !== undefined) setHookV2(config.hook_v2)
      if (config.silence_trim !== undefined) setSilenceTrim(config.silence_trim)
      if (config.use_dlp_subs !== undefined) setUseDlpSubs(config.use_dlp_subs)
      if (config.no_subs !== undefined) setNoSubs(config.no_subs)
      if (config.render_gpus !== undefined) setRenderGpus(config.render_gpus)
      if (config.render_workers_per_gpu !== undefined) setRenderWorkersPerGpu(config.render_workers_per_gpu)
      
      // Default to true when cloning to save AI tokens, user can untoggle
      setLoadGeminiJson(true)
    }
  }, [location.state])

  const handleDiscover = async (e) => {
    e.preventDefault()
    if (!discoverQuery.trim()) return
    setDiscovering(true)
    setDiscoverError('')
    setDiscoverResults(null)
    try {
      const result = await discoverVideos(discoverQuery.trim(), 10)
      setDiscoverResults(result.candidates)
    } catch (err) {
      setDiscoverError(err.message)
    } finally {
      setDiscovering(false)
    }
  }

  const handleUseCandidate = (candidate) => {
    setUrl(candidate.url)
    setSource('youtube')
    setMode('url')
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    setError('')
    try {
      const result = await uploadVideo(file)
      setUploadFilename(result.filename)
      setMode('upload')
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    if (mode === 'url' && !url.trim()) {
      setError('URL tidak boleh kosong')
      return
    }
    if (mode === 'upload' && !uploadFilename) {
      setError('Silakan upload video terlebih dahulu')
      return
    }
    if (mode === 'reuse' && !reuseJobId.trim()) {
      setError('Job ID tidak boleh kosong')
      return
    }
    if (mode === 'discover') {
      setError('Pick a candidate from the discovery results first (click "Use this video")')
      return
    }

    setSubmitting(true)
    try {
      const payload = {
        ...(mode === 'url' ? { url: url.trim() } : {}),
        ...(mode === 'upload' ? { upload_filename: uploadFilename } : {}),
        source,
        clips,
        ratio,
        font_style: fontStyle,
        whisper_model: whisperModel,
        whisper_device: whisperDevice,
        ai_provider: aiProvider,
        use_broll: useBroll,
        use_hook_glitch: useHookGlitch,
        use_auto_bgm: useBgm,
        use_karaoke_effect: useKaraoke,
        no_subs: noSubs,
        hook_v2: hookV2,
        silence_trim: silenceTrim,
        use_dlp_subs: useDlpSubs,
        load_gemini_json: loadGeminiJson,
        render_gpus: renderGpus,
        render_workers_per_gpu: renderWorkersPerGpu,
        ...(reuseJobId.trim() ? { reuse_job_id: reuseJobId.trim() } : {}),
        // Manual mode: just fetch/confirm the source video and stop —
        // JobDetail shows the timestamp picker once this completes, and
        // *that* screen submits the real 'manual' render job.
        ...(clipMode === 'manual' ? { mode: 'download_only' } : {}),
      }

      const job = await createJob(payload)
      navigate(`/job/${job.id}`)
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="fade-in">
      <div className="page-header">
        <div>
          <h2>New Clipping Job</h2>
          <p>Generate viral short clips dari video panjang</p>
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        {/* Clip Selection Mode */}
        <div className="card" style={{ marginBottom: '16px' }}>
          <h3 className="card-title" style={{ marginBottom: '16px' }}>🎬 Clip Selection</h3>
          <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
            <button
              type="button"
              className={`btn ${clipMode === 'auto' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
              onClick={() => setClipMode('auto')}
            >
              🤖 AI Auto-Select
            </button>
            <button
              type="button"
              className={`btn ${clipMode === 'manual' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
              onClick={() => setClipMode('manual')}
            >
              ✋ Manual Timestamps
            </button>
          </div>
          <p className="form-hint">
            {clipMode === 'auto'
              ? 'AI reads the transcript and picks the most viral moments for you.'
              : "We'll download the video, then you scrub through it and mark the exact in/out points yourself."}
          </p>
        </div>

        {/* Source Selection */}
        <div className="card" style={{ marginBottom: '16px' }}>
          <h3 className="card-title" style={{ marginBottom: '16px' }}>📥 Video Source</h3>

          {/* Mode toggle */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
            <button
              type="button"
              className={`btn ${mode === 'url' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
              onClick={() => setMode('url')}
            >
              🔗 From URL
            </button>
            <button
              type="button"
              className={`btn ${mode === 'upload' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
              onClick={() => setMode('upload')}
            >
              📁 Upload File
            </button>
            <button
              type="button"
              className={`btn ${mode === 'reuse' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
              onClick={() => setMode('reuse')}
            >
              🔁 Reuse Job
            </button>
            <button
              type="button"
              className={`btn ${mode === 'discover' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
              onClick={() => setMode('discover')}
            >
              🔎 Auto-Discover
            </button>
          </div>

          {mode === 'discover' && (
            <div className="form-group">
              <label className="form-label">Niche / Topic Keywords</label>
              <div style={{ display: 'flex', gap: '8px' }}>
                <input
                  className="form-input"
                  type="text"
                  placeholder="e.g. AI coding tips, budget travel india, 10 minute meals"
                  value={discoverQuery}
                  onChange={(e) => setDiscoverQuery(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleDiscover(e) }}
                />
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={handleDiscover}
                  disabled={discovering || !discoverQuery.trim()}
                >
                  {discovering ? <><span className="spinner"></span> Searching...</> : '🔎 Search'}
                </button>
              </div>
              <p className="form-hint">
                Scores each result 0-25 against optimizeclip.md's source-video checklist
                (topic strength, view momentum, duration fit, freshness, benefit-driven title) —
                no manual research needed. Pick a result below to auto-fill the URL.
              </p>

              {discoverError && (
                <div style={{ color: 'var(--error)', fontSize: '13px', marginTop: '8px' }}>⚠️ {discoverError}</div>
              )}

              {discoverResults && discoverResults.length === 0 && (
                <p className="form-hint">No candidates found — try a broader or different query.</p>
              )}

              {discoverResults && discoverResults.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '12px' }}>
                  {discoverResults.map((c) => (
                    <DiscoverCandidateCard key={c.video_id} candidate={c} onUse={() => handleUseCandidate(c)} />
                  ))}
                </div>
              )}
            </div>
          )}

          {mode === 'url' && (
            <>
              <div className="form-group">
                <label className="form-label">Video URL</label>
                <input
                  className="form-input"
                  type="url"
                  placeholder="https://www.youtube.com/watch?v=..."
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                />
                <p className="form-hint">Mendukung YouTube, TikTok, Instagram, Google Drive</p>
              </div>
              <div className="form-group" style={{ maxWidth: '200px' }}>
                <label className="form-label">Platform</label>
                <select className="form-select" value={source} onChange={(e) => setSource(e.target.value)}>
                  <option value="youtube">YouTube</option>
                  <option value="tiktok">TikTok</option>
                  <option value="instagram">Instagram</option>
                  <option value="gdrive">Google Drive</option>
                </select>
              </div>
            </>
          )}

          {mode === 'upload' && (
            <div className="form-group">
              <label className="form-label">Upload Video</label>
              {uploadFilename ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <span style={{ color: 'var(--success)' }}>✅ {uploadFilename}</span>
                  <button type="button" className="btn btn-ghost btn-sm" onClick={() => { setUploadFilename(''); fileRef.current?.click() }}>
                    Change
                  </button>
                </div>
              ) : (
                <div>
                  <input
                    ref={fileRef}
                    type="file"
                    accept="video/*"
                    onChange={handleFileUpload}
                    style={{ display: 'none' }}
                  />
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => fileRef.current?.click()}
                    disabled={uploading}
                  >
                    {uploading ? <><span className="spinner"></span> Uploading...</> : '📁 Select Video File'}
                  </button>
                  <p className="form-hint">MP4, MKV, AVI, MOV, WebM (max 2GB)</p>
                </div>
              )}
            </div>
          )}

          {mode === 'reuse' && (
            <div className="form-group">
              <label className="form-label">Reuse Job ID</label>
              <input
                className="form-input"
                type="text"
                placeholder="Contoh: d20b47341e08"
                value={reuseJobId}
                onChange={(e) => setReuseJobId(e.target.value)}
              />
              <p className="form-hint" style={{ marginTop: '4px' }}>Bypass download dengan job ID lama. (Jika menggunakan Clone & Rerun, biarkan form ini terisi).</p>
            </div>
          )}
        </div>

        {/* Main Config */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px', marginBottom: '16px' }}>
          {/* Basic Settings */}
          <div className="config-section">
            <h4>🎯 Basic Settings</h4>
            {clipMode === 'auto' && (
              <div className="form-group">
                <label className="form-label">Number of Clips</label>
                <input className="form-input" type="number" min="1" max="30" value={clips} onChange={(e) => setClips(parseInt(e.target.value) || 7)} />
              </div>
            )}
            <div className="form-group">
              <label className="form-label">Aspect Ratio</label>
              <select className="form-select" value={ratio} onChange={(e) => setRatio(e.target.value)}>
                <option value="9:16">9:16 (Vertical — TikTok/Reels)</option>
                <option value="16:9">16:9 (Horizontal)</option>
                <option value="1:1">1:1 (Square)</option>
                <option value="3:4">3:4</option>
                <option value="4:5">4:5</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Font Style</label>
              <select className="form-select" value={fontStyle} onChange={(e) => setFontStyle(e.target.value)}>
                <option value="HORMOZI">Hormozi (Bold)</option>
                <option value="DEFAULT">Default (Montserrat)</option>
                <option value="STORYTELLER">Storyteller (Inter)</option>
                <option value="CINEMATIC">Cinematic (Bebas Neue)</option>
              </select>
            </div>
          </div>

          {/* AI Settings */}
          <div className="config-section">
            <h4>🤖 AI & Whisper</h4>
            {clipMode === 'auto' && (
              <div className="form-group">
                <label className="form-label">AI Provider</label>
                <select className="form-select" value={aiProvider} onChange={(e) => setAiProvider(e.target.value)}>
                  <option value="gemini">Google Gemini</option>
                  <option value="nvidia">NVIDIA NIM</option>
                </select>
              </div>
            )}
            <div className="form-group">
              <label className="form-label">Whisper Model</label>
              <select className="form-select" value={whisperModel} onChange={(e) => setWhisperModel(e.target.value)}>
                <option value="large-v3">large-v3 (Best quality)</option>
                <option value="medium">medium (Balanced)</option>
                <option value="small">small (Fast)</option>
                <option value="base">base (Fastest)</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Device</label>
              <select className="form-select" value={whisperDevice} onChange={(e) => setWhisperDevice(e.target.value)}>
                <option value="cuda">CUDA (GPU)</option>
                <option value="cpu">CPU</option>
                <option value="auto">Auto</option>
              </select>
            </div>
          </div>

          {/* Feature Toggles */}
          <div className="config-section">
            <h4>✨ Features</h4>
            <ToggleRow label="B-Roll Footage" desc="Insert stock footage" checked={useBroll} onChange={setUseBroll} />
            <ToggleRow label="Hook Glitch" desc="Glitch transition intro" checked={useHookGlitch} onChange={setUseHookGlitch} />
            <ToggleRow label="Background Music" desc="Auto BGM matching" checked={useBgm} onChange={setUseBgm} />
            <ToggleRow label="Karaoke Effect" desc="Word-by-word highlight" checked={useKaraoke} onChange={setUseKaraoke} />
            {clipMode === 'auto' && (
              <>
                <ToggleRow label="Hook V2" desc="Multi-hook intro clips" checked={hookV2} onChange={setHookV2} />
                <ToggleRow label="Silence Trim" desc="Remove dead air" checked={silenceTrim} onChange={setSilenceTrim} />
              </>
            )}
            <ToggleRow label="YouTube Subs" desc="Skip Whisper if available" checked={useDlpSubs} onChange={setUseDlpSubs} />
            <ToggleRow label="No Subtitles" desc="Render without text" checked={noSubs} onChange={setNoSubs} />
            {clipMode === 'auto' && (
              <ToggleRow label="Bypass AI" desc="Reuse gemini JSON (if exist)" checked={loadGeminiJson} onChange={setLoadGeminiJson} />
            )}
          </div>

          {/* GPU Rendering */}
          <div className="config-section">
            <h4>🖥️ GPU Rendering</h4>
            <div className="form-group">
              <label className="form-label">GPUs to use</label>
              <select className="form-select" value={renderGpus} onChange={(e) => setRenderGpus(e.target.value)}>
                <option value="auto">Auto (use every visible GPU)</option>
                <option value="1">1 GPU only</option>
                <option value="2">2 GPUs</option>
              </select>
              <p className="form-hint">This job's clips are split across all selected GPUs and rendered in parallel.</p>
            </div>
            <div className="form-group">
              <label className="form-label">Render workers per GPU</label>
              <input
                className="form-input"
                type="number"
                min="1"
                max="4"
                value={renderWorkersPerGpu}
                onChange={(e) => setRenderWorkersPerGpu(parseInt(e.target.value) || 1)}
              />
              <p className="form-hint">Keep at 1 unless you've benchmarked that more processes per GPU actually increases clips/hour — NVENC/VRAM contention can make it worse.</p>
            </div>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div style={{ background: 'var(--error-dim)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 'var(--radius-md)', padding: '12px 16px', marginBottom: '16px', color: 'var(--error)', fontSize: '13px' }}>
            ⚠️ {error}
          </div>
        )}

        {/* Submit */}
        <button type="submit" className="btn btn-primary" disabled={submitting} style={{ fontSize: '14px', padding: '12px 28px' }}>
          {submitting
            ? <><span className="spinner"></span> Processing...</>
            : clipMode === 'manual' ? '⬇️ Download Video & Pick Clips' : '🚀 Start Clipping'}
        </button>
      </form>
    </div>
  )
}

const DISCOVER_LABELS = {
  high_priority: { text: '🟢 High Priority', color: 'var(--success)' },
  okay: { text: '🟡 Okay', color: 'var(--warning, #eab308)' },
  skip: { text: '🔴 Skip', color: 'var(--error)' },
}

function DiscoverCandidateCard({ candidate, onUse }) {
  const label = DISCOVER_LABELS[candidate.label] || DISCOVER_LABELS.skip
  const minutes = (candidate.duration_sec / 60).toFixed(1)

  return (
    <div className="card" style={{ padding: '12px 16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', alignItems: 'flex-start' }}>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <span style={{ fontWeight: 700, fontSize: '13px', color: label.color }}>{label.text}</span>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>score {candidate.score}/25</span>
          </div>
          <div style={{ fontSize: '14px', fontWeight: 600, marginTop: '4px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {candidate.title}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
            {candidate.channel} · {candidate.view_count.toLocaleString()} views · {minutes} min
            {candidate.upload_date && ` · ${candidate.upload_date.slice(0,4)}-${candidate.upload_date.slice(4,6)}-${candidate.upload_date.slice(6,8)}`}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px' }}>
            {Object.entries(candidate.breakdown).map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v}`).join(' · ')}
          </div>
          {candidate.reasons?.length > 0 && (
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px', fontStyle: 'italic' }}>
              {candidate.reasons[0]}
            </div>
          )}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', flexShrink: 0 }}>
          <button type="button" className="btn btn-primary btn-sm" onClick={onUse}>Use this video</button>
          <a href={candidate.url} target="_blank" rel="noreferrer" className="btn btn-ghost btn-sm" style={{ textAlign: 'center' }}>
            Preview ↗
          </a>
        </div>
      </div>
    </div>
  )
}

function ToggleRow({ label, desc, checked, onChange }) {
  return (
    <div className="toggle-row">
      <div>
        <div className="toggle-label">{label}</div>
        {desc && <div className="toggle-desc">{desc}</div>}
      </div>
      <label className="toggle-switch">
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span className="toggle-slider"></span>
      </label>
    </div>
  )
}

export default NewJob
