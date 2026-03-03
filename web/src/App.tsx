import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'

const THEME_KEY = 'eval-dashboard-theme'

function getWindowLabels(data: EvalResult | null): number[] {
  if (!data?.predictions?.window_labels) return []
  const w = data.predictions.window_labels
  return Array.isArray(w) ? w : []
}

interface EvalResult {
  num_windows?: number
  predictions?: { window_labels?: number[] }
  metrics?: unknown
  [k: string]: unknown
}

export default function App() {
  const [dark, setDark] = useState(() => {
    if (typeof window === 'undefined') return false
    return localStorage.getItem(THEME_KEY) === 'dark'
  })
  const [data, setData] = useState<EvalResult | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [selectionStart, setSelectionStart] = useState<number | null>(null)
  const [selectionEnd, setSelectionEnd] = useState<number | null>(null)

  const numWindows = data?.num_windows ?? getWindowLabels(data).length ?? 0
  const windowLabels = getWindowLabels(data)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light')
    localStorage.setItem(THEME_KEY, dark ? 'dark' : 'light')
  }, [dark])

  const onFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      try {
        const json = JSON.parse(ev.target?.result as string) as EvalResult
        setData(json)
        setLoadError(null)
        setSelectionStart(null)
        setSelectionEnd(null)
      } catch (err) {
        setLoadError(err instanceof Error ? err.message : 'Invalid JSON')
        setData(null)
      }
      e.target.value = ''
    }
    reader.readAsText(file)
  }, [])

  const summaryPreview = data
    ? JSON.stringify(data, null, 2).slice(0, 8000) +
      (JSON.stringify(data).length > 8000 ? '\n… (truncated)' : '')
    : ''

  const onCellClick = useCallback((index: number) => {
    setSelectionStart((prev) => {
      if (prev == null) {
        setSelectionEnd(index)
        return index
      }
      setSelectionEnd(index)
      return prev
    })
  }, [])

  const clearSelection = useCallback(() => {
    setSelectionStart(null)
    setSelectionEnd(null)
  }, [])

  const lo = selectionStart != null && selectionEnd != null ? Math.min(selectionStart, selectionEnd) : null
  const hi = selectionStart != null && selectionEnd != null ? Math.max(selectionStart, selectionEnd) : null
  const subsetCount = lo != null && hi != null ? hi - lo + 1 : 0

  return (
    <>
      <header className="header">
        <h1>Eval Dashboard</h1>
        <div className="header-actions">
          <label className="theme-toggle">
            <input
              type="checkbox"
              checked={dark}
              onChange={(e) => setDark(e.target.checked)}
              aria-label="Dark mode"
            />
            <span className="toggle-slider" />
            <span className="toggle-label">Dark</span>
          </label>
          <div className="file-picker">
            <label htmlFor="dataFile">Load result</label>
            <input type="file" id="dataFile" accept=".json" onChange={onFileChange} />
          </div>
        </div>
      </header>

      <main className="main">
        <section className="summary" aria-live="polite">
          {loadError && <p className="summary-placeholder summary-error">{loadError}</p>}
          {!data && !loadError && (
            <p className="summary-placeholder">Load a result JSON (eval or comparison) to see metrics.</p>
          )}
          {data && !loadError && (
            <pre className="summary-json">{summaryPreview}</pre>
          )}
        </section>

        <section className="timeline-section">
          <div className="timeline-header">
            <h2>Windows timeline</h2>
            <div className="timeline-info">
              <span>{numWindows} window{numWindows !== 1 ? 's' : ''}</span>
              <span className="timeline-hint">
                Scroll with trackpad or use ← → keys. Click to select range.
              </span>
            </div>
          </div>
          <Timeline
            numWindows={numWindows}
            windowLabels={windowLabels}
            selectionStart={selectionStart}
            selectionEnd={selectionEnd}
            onCellClick={onCellClick}
          />
        </section>

        <section className="subset-section">
          <h2>Subset</h2>
          <p className="subset-desc">
            Select a range on the timeline (click start then end). Each window is 300 timesteps;
            analyzing a subset runs the same model on those windows only — no change to input size.
          </p>
          <div className="subset-info">
            <span>
              {lo != null && hi != null ? `Windows ${lo} – ${hi}` : 'No selection'}
            </span>
            {subsetCount > 0 && (
              <span>{subsetCount} window{subsetCount !== 1 ? 's' : ''}</span>
            )}
          </div>
          <div className="subset-actions">
            <button
              type="button"
              onClick={clearSelection}
              disabled={selectionStart == null}
            >
              Clear selection
            </button>
          </div>
        </section>
      </main>
    </>
  )
}

const CELL_WIDTH_PX = 12

function Timeline({
  numWindows,
  windowLabels,
  selectionStart,
  selectionEnd,
  onCellClick,
}: {
  numWindows: number
  windowLabels: number[]
  selectionStart: number | null
  selectionEnd: number | null
  onCellClick: (index: number) => void
}) {
  const wrapperRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const thumbRef = useRef<HTMLDivElement>(null)
  const [thumbStyle, setThumbStyle] = useState<{ width: number; left: number } | null>(null)
  const [showScrollbar, setShowScrollbar] = useState(false)

  useEffect(() => {
    const wrap = wrapperRef.current
    const trackW = numWindows * CELL_WIDTH_PX
    const viewW = wrap?.clientWidth ?? 0
    if (trackW <= viewW) {
      setShowScrollbar(false)
      return
    }
    setShowScrollbar(true)
    const maxScroll = trackW - viewW

    const updateThumb = () => {
      if (!wrap || !scrollRef.current) return
      const pct = maxScroll > 0 ? wrap.scrollLeft / maxScroll : 0
      const barW = wrap.offsetWidth
      const thumbW = Math.max(40, (viewW / trackW) * barW)
      const left = pct * (barW - thumbW)
      setThumbStyle({ width: thumbW, left })
    }

    wrap?.addEventListener('scroll', updateThumb)
    updateThumb()
    return () => wrap?.removeEventListener('scroll', updateThumb)
  }, [numWindows])

  const onWheel = useCallback((e: React.WheelEvent) => {
    if (e.deltaX !== 0) {
      wrapperRef.current?.scrollBy({ left: e.deltaX })
      e.preventDefault()
    }
  }, [])

  const onKeyDown = useCallback((e: React.KeyboardEvent) => {
    const step = 3 * CELL_WIDTH_PX
    if (e.key === 'ArrowLeft') {
      wrapperRef.current?.scrollBy({ left: -step })
      e.preventDefault()
    } else if (e.key === 'ArrowRight') {
      wrapperRef.current?.scrollBy({ left: step })
      e.preventDefault()
    }
  }, [])

  const selectionLeft =
    selectionStart != null && selectionEnd != null
      ? Math.min(selectionStart, selectionEnd) * CELL_WIDTH_PX
      : 0
  const selectionWidth =
    selectionStart != null && selectionEnd != null
      ? (Math.abs(selectionEnd - selectionStart) + 1) * CELL_WIDTH_PX
      : 0

  return (
    <>
      <div
        className="timeline-wrapper"
        ref={wrapperRef}
        onWheel={onWheel}
        style={{ overflowX: 'auto', overflowY: 'hidden', marginBottom: 8, WebkitOverflowScrolling: 'touch' }}
      >
        <div
          className="timeline-scroll"
          ref={scrollRef}
          tabIndex={0}
          role="slider"
          aria-valuemin={0}
          aria-valuemax={Math.max(0, numWindows - 1)}
          aria-valuenow={0}
          onKeyDown={onKeyDown}
        >
          <div className="timeline-track" style={{ width: numWindows * CELL_WIDTH_PX }}>
            {Array.from({ length: numWindows }, (_, i) => {
              const label = windowLabels[i]
              const cls =
                label != null && label !== 0 ? 'fault' : label != null && label === 0 ? 'ok' : 'neutral'
              return (
                <div
                  key={i}
                  className={`timeline-cell ${cls}`}
                  onClick={() => onCellClick(i)}
                  role="button"
                  tabIndex={-1}
                  aria-label={`Window ${i}${label != null ? ` label ${label}` : ''}`}
                  style={{ width: CELL_WIDTH_PX, minWidth: CELL_WIDTH_PX }}
                />
              )
            })}
          </div>
          <div
            className="timeline-selection"
            style={{ left: selectionLeft, width: selectionWidth }}
          />
        </div>
      </div>
      {showScrollbar && (
        <div
          className="timeline-scrollbar"
          onClick={(e) => {
            if (e.target === thumbRef.current) return
            const wrap = wrapperRef.current
            const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
            const x = e.clientX - rect.left
            const pct = x / rect.width
            const trackW = numWindows * CELL_WIDTH_PX
            const viewW = wrap?.clientWidth ?? 0
            wrap?.scrollTo({ left: pct * (trackW - viewW) })
          }}
        >
          <div className="scrollbar-track">
            <div
              ref={thumbRef}
              className="scrollbar-thumb"
              style={
                thumbStyle
                  ? { width: thumbStyle.width, left: thumbStyle.left }
                  : undefined
              }
              onMouseDown={(e) => {
                e.preventDefault()
                const startX = e.clientX
                const startScroll = wrapperRef.current?.scrollLeft ?? 0
                const trackW = numWindows * CELL_WIDTH_PX
                const viewW = wrapperRef.current?.clientWidth ?? 0
                const maxScroll = trackW - viewW
                const barW = (e.currentTarget.parentElement?.parentElement as HTMLElement)?.offsetWidth ?? 1
                const thumbW = thumbRef.current?.offsetWidth ?? 40
                const scale = maxScroll / Math.max(1, barW - thumbW)
                const move = (ev: MouseEvent) => {
                  const dx = ev.clientX - startX
                  wrapperRef.current?.scrollTo({
                    left: Math.max(0, Math.min(maxScroll, startScroll + dx * scale)),
                  })
                }
                const up = () => {
                  document.removeEventListener('mousemove', move)
                  document.removeEventListener('mouseup', up)
                }
                document.addEventListener('mousemove', move)
                document.addEventListener('mouseup', up)
              }}
            />
          </div>
        </div>
      )}
    </>
  )
}
