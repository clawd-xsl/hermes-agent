import { PassThrough } from 'stream'

import { renderSync, type ScrollBoxHandle } from '@hermes/ink'
import React, { useRef, useState } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { patchUiState } from '../app/uiStore.js'
import { applyReloadedHistory, useHistorySync } from '../app/useHistorySync.js'
import type { Msg } from '../types.js'

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

const intro: Msg = { info: { model: 'm' } as Msg['info'], kind: 'intro', role: 'system', text: '' }
const old: Msg[] = [intro, { role: 'user', text: 'hello' }, { role: 'assistant', text: 'hi' }]

const reloadedMessages = [
  { role: 'user', text: 'hello' },
  { role: 'assistant', text: 'hi' },
  { role: 'user', text: 'sent via signal' },
  { role: 'assistant', text: 'answered on signal' }
]

type Reply = Record<string, unknown>

function Harness({
  busy,
  calls,
  expose,
  replies,
  rotated
}: {
  busy: boolean
  calls: string[]
  expose: { items: Msg[] }
  replies: Reply[]
  rotated: string[]
}) {
  const [items, setItems] = useState<Msg[]>(old)
  const scrollRef = useRef<ScrollBoxHandle | null>(null)

  expose.items = items

  const gw = {
    request: async <T>(method: string) => {
      calls.push(method)

      return (replies.shift() ?? { reloaded: false, reason: 'unchanged' }) as T
    }
  }

  useHistorySync({
    busy,
    gw,
    intervalMs: 20,
    onRotated: id => rotated.push(id),
    scrollRef,
    setHistoryItems: setItems,
    sid: 'sid-1'
  })

  return null
}

const mount = (props: Omit<React.ComponentProps<typeof Harness>, 'expose'> & { expose: { items: Msg[] } }) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 20 })
  Object.assign(stdin, { isTTY: false })
  stdout.on('data', () => {})

  return renderSync(React.createElement(Harness, props), {
    exitOnCtrlC: false,
    patchConsole: false,
    stderr,
    stdin,
    stdout
  } as Parameters<typeof renderSync>[1])
}

describe('applyReloadedHistory', () => {
  it('replaces the transcript but keeps the local intro banner', () => {
    const next = applyReloadedHistory(old, reloadedMessages)

    expect(next[0]).toBe(intro)
    expect(next.map(m => m.text).slice(1)).toEqual(['hello', 'hi', 'sent via signal', 'answered on signal'])
  })
})

describe('useHistorySync', () => {
  afterEach(() => patchUiState({ busy: false, sid: null }))

  it('shows turns another writer persisted while the session sits idle', async () => {
    patchUiState({ busy: false, sid: 'sid-1' })
    const expose = { items: [] as Msg[] }
    const calls: string[] = []

    const replies: Reply[] = [
      { reason: 'baseline', reloaded: false },
      { messages: reloadedMessages, reloaded: true, session_id: 'sid-1' }
    ]

    const instance = mount({ busy: false, calls, expose, replies, rotated: [] })

    await delay(120)
    instance.unmount()

    expect(calls.every(m => m === 'session.history.reload')).toBe(true)
    expect(expose.items.map(m => m.text)).toContain('answered on signal')
    expect(expose.items[0]?.kind).toBe('intro')
  })

  it('never polls while a turn is running', async () => {
    patchUiState({ busy: true, sid: 'sid-1' })
    const calls: string[] = []
    const instance = mount({ busy: true, calls, expose: { items: [] }, replies: [], rotated: [] })

    await delay(80)
    instance.unmount()

    expect(calls).toEqual([])
  })

  it('hands a compression continuation to the resume path', async () => {
    patchUiState({ busy: false, sid: 'sid-1' })
    const rotated: string[] = []

    const instance = mount({
      busy: false,
      calls: [],
      expose: { items: [] },
      replies: [{ reloaded: false, rotated_to: 'tip-session' }],
      rotated
    })

    await delay(60)
    instance.unmount()

    expect(rotated).toEqual(['tip-session'])
  })
})
