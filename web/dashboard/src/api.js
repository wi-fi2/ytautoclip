const API_BASE = '/api'

export async function fetchJobs() {
  const res = await fetch(`${API_BASE}/jobs`)
  if (!res.ok) throw new Error('Failed to fetch jobs')
  return res.json()
}

export async function fetchJob(jobId) {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`)
  if (!res.ok) throw new Error('Job not found')
  return res.json()
}

export async function createJob(payload) {
  const res = await fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to create job')
  }
  return res.json()
}

export async function deleteJob(jobId) {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to delete job')
  return res.json()
}

export async function uploadVideo(file, onProgress) {
  const formData = new FormData()
  formData.append('file', file)

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Upload failed')
  }
  return res.json()
}

export async function fetchSettings() {
  const res = await fetch(`${API_BASE}/settings`)
  if (!res.ok) throw new Error('Failed to fetch settings')
  return res.json()
}

export async function updateSettings(payload) {
  const res = await fetch(`${API_BASE}/settings`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error('Failed to update settings')
  return res.json()
}

export async function discoverVideos(query, limit = 10) {
  const params = new URLSearchParams({ query, limit: String(limit) })
  const res = await fetch(`${API_BASE}/discover?${params}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Discovery search failed')
  }
  return res.json()
}

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw new Error('Health check failed')
  return res.json()
}

export async function fetchCloudStatus() {
  const res = await fetch(`${API_BASE}/cloud/status`)
  if (!res.ok) throw new Error('Failed to fetch cloud status')
  return res.json()
}

export async function fetchCloudBacklog() {
  const res = await fetch(`${API_BASE}/cloud/backlog`)
  if (!res.ok) throw new Error('Failed to fetch backlog')
  return res.json()
}

export async function addCloudBacklog(urls) {
  const res = await fetch(`${API_BASE}/cloud/backlog`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ urls }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to add to backlog')
  }
  return res.json()
}

export async function clearCloudBacklog() {
  const res = await fetch(`${API_BASE}/cloud/backlog`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to clear backlog')
  return res.json()
}

export async function startCloudPrep(payload) {
  const res = await fetch(`${API_BASE}/cloud/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to start prep loop')
  }
  return res.json()
}

export async function stopCloudPrep() {
  const res = await fetch(`${API_BASE}/cloud/stop`, { method: 'POST' })
  if (!res.ok) throw new Error('Failed to stop prep loop')
  return res.json()
}

export function createSSEConnection(jobId, onMessage) {
  const eventSource = new EventSource(`${API_BASE}/jobs/${jobId}/status`)

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
    } catch (e) {
      console.error('SSE parse error:', e)
    }
  }

  eventSource.onerror = () => {
    eventSource.close()
  }

  return eventSource
}
