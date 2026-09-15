import { useState, useEffect, useRef, useCallback } from 'react'
import {
  fetchCloudStatus,
  fetchCloudBacklog,
  addCloudBacklog,
  clearCloudBacklog,
  startCloudPrep,
  stopCloudPrep,
} from '../api'

const STATUS_COLORS = {
  READY: 'var(--text-secondary)',
  SENT: 'var(--accent)',
  COMPLETED: 'var(--success)',
  FAILED: 'var(--error)',
}

function CloudBatch() {
  const [status, setStatus] = useState(null)
  const [backlog, setBacklog] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [msg, setMsg] = useState('')

  const [urlsText, setUrlsText] = useState('')
  const [adding, setAdding] = useState(false)

  const [inboxDataset, setInboxDataset] = useState('')
  const [outboxDataset, setOutboxDataset] = useState('')
  const [thresholdClips, setThresholdClips] = useState(60)
  const [prepWorkers, setPrepWorkers] = useState(2)
  const [starting, setStarting] = useState(false)
  const [stopping, setStopping] = useState(false)

  const pollRef = useRef(null)

  const refresh = useCallback(async () => {
    try {
      const [s, b] = await Promise.all([fetchCloudStatus(), fetchCloudBacklog()])
      setStatus(s)
      setBacklog(b.urls)
      setError('')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    pollRef.current = setInterval(refresh, 5000)
    return () => clearInterval(pollRef.current)
  }, [refresh])

  const handleAddUrls = async (e) => {
    e.preventDefault()
    const urls = urlsText.split('\n').map((u) => u.trim()).filter(Boolean)
    if (urls.length === 0) return
    setAdding(true)
    setMsg('')
    try {
      await addCloudBacklog(urls)
      setUrlsText('')
      await refresh()
      setMsg(`✅ Added ${urls.length} URL(s) to backlog`)
    } catch (err) {
      setError(err.message)
    } finally {
      setAdding(false)
    }
  }

  const handleClearBacklog = async () => {
    if (!confirm('Clear the entire backlog?')) return
    try {
      await clearCloudBacklog()
      await refresh()
    } catch (err) {
      setError(err.message)
    }
  }

  const handleStart = async (e) => {
    e.preventDefault()
    if (!inboxDataset.trim() || !outboxDataset.trim()) {
      setError('Inbox and outbox dataset ids are required (e.g. yourname/ytclipper-inbox).')
      return
    }
    setStarting(true)
    setError('')
    setMsg('')
    try {
      await startCloudPrep({
        inbox_dataset: inboxDataset.trim(),
        outbox_dataset: outboxDataset.trim(),
        threshold_clips: thresholdClips,
        prep_workers: prepWorkers,
      })
      await refresh()
      setMsg('🚀 Prep loop started')
    } catch (err) {
      setError(err.message)
    } finally {
      setStarting(false)
    }
  }

  const handleStop = async () => {
    setStopping(true)
    try {
      await stopCloudPrep()
      await refresh()
      setMsg('🛑 Prep loop stopped')
    } catch (err) {
      setError(err.message)
    } finally {
      setStopping(false)
    }
  }

  if (loading) return <div className="empty-state"><div className="spinner"></div></div>

  const counts = status?.job_counts || {}
  const clipsReady = counts.READY?.clips || 0
  const thresholdHit = status?.threshold_clips && clipsReady >= status.threshold_clips

  return (
    <div className="fade-in">
      <div className="page-header">
        <div>
          <h2>☁️ Cloud Batch</h2>
          <p>PC prep loop + Kaggle dual-GPU render, connected via a Kaggle Dataset mailbox (no GCS/Drive needed)</p>
        </div>
      </div>

      {error && (
        <div style={{ background: 'var(--error-dim)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 'var(--radius-md)', padding: '12px 16px', marginBottom: '16px', color: 'var(--error)', fontSize: '13px' }}>
          ⚠️ {error}
        </div>
      )}
      {msg && (
        <div style={{ fontSize: '13px', color: 'var(--success)', marginBottom: '16px' }}>{msg}</div>
      )}

      {/* Status */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <h3 className="card-title">Prep loop status</h3>
          <span style={{
            fontWeight: 700, fontSize: '13px',
            color: status?.running ? 'var(--success)' : 'var(--text-tertiary)',
          }}>
            {status?.running ? `🟢 Running (PID ${status.pid})` : '⚪ Stopped'}
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '12px', marginBottom: '16px' }}>
          {['READY', 'SENT', 'COMPLETED', 'FAILED'].map((s) => (
            <div key={s} style={{ padding: '10px 12px', borderRadius: 'var(--radius-md)', background: 'var(--bg-secondary)' }}>
              <div style={{ fontSize: '11px', color: STATUS_COLORS[s], fontWeight: 700 }}>{s}</div>
              <div style={{ fontSize: '20px', fontWeight: 700 }}>{counts[s]?.videos || 0}</div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>{counts[s]?.clips || 0} clips</div>
            </div>
          ))}
        </div>

        {status?.running && (
          <>
            <div style={{ fontSize: '13px', color: thresholdHit ? 'var(--success)' : 'var(--text-secondary)', marginBottom: '8px' }}>
              {thresholdHit
                ? `🚀 Threshold reached (${clipsReady}/${status.threshold_clips} clips) — a batch push may already be in flight. Start your Kaggle notebook if it isn't running.`
                : `⏳ ${clipsReady}/${status.threshold_clips} clips ready — waiting for enough backlog before pushing to Kaggle.`}
            </div>
            <button type="button" className="btn btn-secondary btn-sm" onClick={handleStop} disabled={stopping}>
              {stopping ? <><span className="spinner"></span> Stopping...</> : '🛑 Stop prep loop'}
            </button>
          </>
        )}

        {status?.recent_log?.length > 0 && (
          <details style={{ marginTop: '16px' }}>
            <summary style={{ cursor: 'pointer', fontSize: '13px', color: 'var(--text-secondary)' }}>📜 Recent log ({status.recent_log.length} lines)</summary>
            <pre style={{
              marginTop: '8px', maxHeight: '260px', overflowY: 'auto', fontSize: '12px',
              background: 'var(--bg-secondary)', padding: '12px', borderRadius: 'var(--radius-md)',
              whiteSpace: 'pre-wrap', wordBreak: 'break-word',
            }}>
              {status.recent_log.join('\n')}
            </pre>
          </details>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
        {/* Backlog */}
        <div className="card">
          <h3 className="card-title" style={{ marginBottom: '12px' }}>📋 Backlog ({backlog.length})</h3>
          <form onSubmit={handleAddUrls}>
            <div className="form-group">
              <label className="form-label">Add source URLs (one per line)</label>
              <textarea
                className="form-input"
                rows={5}
                placeholder="https://www.youtube.com/watch?v=...&#10;https://www.youtube.com/watch?v=..."
                value={urlsText}
                onChange={(e) => setUrlsText(e.target.value)}
              />
            </div>
            <button type="submit" className="btn btn-secondary btn-sm" disabled={adding || !urlsText.trim()}>
              {adding ? <><span className="spinner"></span> Adding...</> : '➕ Add to backlog'}
            </button>
          </form>

          {backlog.length > 0 && (
            <>
              <div style={{ marginTop: '16px', maxHeight: '200px', overflowY: 'auto', fontSize: '12px', color: 'var(--text-secondary)' }}>
                {backlog.map((u, i) => <div key={i} style={{ padding: '4px 0', borderBottom: '1px solid var(--border-color)' }}>{u}</div>)}
              </div>
              <button type="button" className="btn btn-ghost btn-sm" onClick={handleClearBacklog} style={{ marginTop: '12px' }}>
                🗑️ Clear backlog
              </button>
            </>
          )}
        </div>

        {/* Start prep loop */}
        <div className="card">
          <h3 className="card-title" style={{ marginBottom: '12px' }}>🚀 Kaggle dataset config</h3>
          <form onSubmit={handleStart}>
            <div className="form-group">
              <label className="form-label">Inbox dataset</label>
              <input
                className="form-input" type="text" placeholder="yourname/ytclipper-inbox"
                value={inboxDataset} onChange={(e) => setInboxDataset(e.target.value)}
                disabled={status?.running}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Outbox dataset</label>
              <input
                className="form-input" type="text" placeholder="yourname/ytclipper-outbox"
                value={outboxDataset} onChange={(e) => setOutboxDataset(e.target.value)}
                disabled={status?.running}
              />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div className="form-group">
                <label className="form-label">Threshold (clips)</label>
                <input
                  className="form-input" type="number" min="1"
                  value={thresholdClips} onChange={(e) => setThresholdClips(parseInt(e.target.value) || 60)}
                  disabled={status?.running}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Prep workers</label>
                <input
                  className="form-input" type="number" min="1" max="8"
                  value={prepWorkers} onChange={(e) => setPrepWorkers(parseInt(e.target.value) || 2)}
                  disabled={status?.running}
                />
              </div>
            </div>
            <p className="form-hint">
              A batch is pushed to the inbox dataset once this many clips are queued — that's when it's worth
              starting the Kaggle notebook, so its GPU-hours aren't spent waiting on a thin backlog.
            </p>
            {!status?.running && (
              <button type="submit" className="btn btn-primary" disabled={starting || backlog.length === 0}>
                {starting ? <><span className="spinner"></span> Starting...</> : '▶️ Start prep loop'}
              </button>
            )}
            {backlog.length === 0 && !status?.running && (
              <p className="form-hint" style={{ color: 'var(--error)' }}>Add at least one URL to the backlog first.</p>
            )}
          </form>
        </div>
      </div>

      <div className="card" style={{ marginTop: '16px' }}>
        <h3 className="card-title" style={{ marginBottom: '8px' }}>ℹ️ How this works</h3>
        <ol style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.7, paddingLeft: '20px' }}>
          <li>Add source URLs to the backlog, set your Kaggle dataset ids, and start the prep loop — it runs on this PC, downloading/transcribing/analyzing videos and pushing manifests (not video bytes) to the inbox dataset.</li>
          <li>When the threshold is reached, a batch is pushed and this page will say so — that's your cue to open your Kaggle notebook (T4 x2, internet on, secrets attached) and run <code>clipping.cloud.kaggle_consumer</code>.</li>
          <li>The Kaggle session renders clips across both GPUs and pushes finished clips to the outbox dataset periodically — it keeps polling the inbox for new work for as long as it runs.</li>
          <li>This prep loop pulls the outbox on its own and copies finished clips into <code>outputs/cloud_batches/</code> automatically — no manual download step.</li>
        </ol>
      </div>
    </div>
  )
}

export default CloudBatch
