import { forceRedraw, type ScrollBoxHandle } from '@hermes/ink'
import { type RefObject, useEffect, useRef } from 'react'

import { INLINE_MODE } from '../config/env.js'
import { HISTORY_SYNC_INTERVAL_MS } from '../config/timing.js'
import { toTranscriptMessages } from '../domain/messages.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { SessionHistoryReloadResponse } from '../gatewayTypes.js'
import { asRpcResult } from '../lib/rpc.js'
import type { Msg } from '../types.js'

import type { StateSetter } from './interfaces.js'
import { getUiState } from './uiStore.js'
import { scheduleResumeScrollToBottom } from './useSessionLifecycle.js'

// Inline mode paints into the primary buffer: rows that already scrolled into
// the host terminal's scrollback are never rewritten by the diff renderer, so a
// replaced transcript must be repainted from scratch (screen AND scrollback).
// Delay past the commit + the post-mount height pass so the repaint sees the
// settled frame.
const INLINE_REPAINT_DELAY_MS = 150

/** Replace the transcript with a reloaded one, keeping the local intro banner. */
export const applyReloadedHistory = (prev: readonly Msg[], messages: unknown): Msg[] => {
  const intro = prev[0]?.kind === 'intro' ? [prev[0]] : []

  return [...intro, ...toTranscriptMessages(messages)]
}

export interface HistorySyncOptions {
  busy: boolean
  gw: Pick<GatewayClient, 'request'>
  intervalMs?: number
  onRotated: (sessionId: string) => void
  scrollRef: RefObject<null | ScrollBoxHandle>
  setHistoryItems: StateSetter<Msg[]>
  sid: null | string
  stdout?: NodeJS.WriteStream
}

/**
 * Keep an idle, resumed session in step with turns other writers persisted
 * (e.g. Signal/Telegram turns on the same conversation). The dashboard's
 * keep-alive PTY is reattached on reopen instead of respawned, so without
 * this a reopened chat shows the transcript as of the TUI's spawn forever.
 */
export function useHistorySync({
  busy,
  gw,
  intervalMs = HISTORY_SYNC_INTERVAL_MS,
  onRotated,
  scrollRef,
  setHistoryItems,
  sid,
  stdout
}: HistorySyncOptions) {
  const onRotatedRef = useRef(onRotated)

  onRotatedRef.current = onRotated

  useEffect(() => {
    if (!sid || busy) {
      return
    }

    let disposed = false
    let inFlight = false
    let cancelScroll: (() => void) | undefined
    let repaintTimer: ReturnType<typeof setTimeout> | undefined

    const stillCurrent = () => !disposed && getUiState().sid === sid && !getUiState().busy

    const tick = () => {
      if (inFlight || !stillCurrent()) {
        return
      }

      inFlight = true

      gw.request<SessionHistoryReloadResponse>('session.history.reload', { session_id: sid })
        .then(raw => {
          const r = asRpcResult<SessionHistoryReloadResponse>(raw)

          if (!r || !stillCurrent()) {
            return
          }

          if (!r.reloaded) {
            if (r.rotated_to) {
              onRotatedRef.current(r.rotated_to)
            }

            return
          }

          setHistoryItems(prev => applyReloadedHistory(prev, r.messages))
          cancelScroll?.()
          cancelScroll = scheduleResumeScrollToBottom(scrollRef)

          if (INLINE_MODE) {
            clearTimeout(repaintTimer)
            repaintTimer = setTimeout(
              () => forceRedraw(stdout ?? process.stdout, { clearScrollback: true }),
              INLINE_REPAINT_DELAY_MS
            )
          }
        })
        .catch(() => {})
        .finally(() => {
          inFlight = false
        })
    }

    // First call records the gateway-side baseline right after resume/idle.
    tick()
    const timer = setInterval(tick, intervalMs)

    return () => {
      disposed = true
      clearInterval(timer)
      cancelScroll?.()
      clearTimeout(repaintTimer)
    }
  }, [busy, gw, intervalMs, scrollRef, setHistoryItems, sid, stdout])
}
