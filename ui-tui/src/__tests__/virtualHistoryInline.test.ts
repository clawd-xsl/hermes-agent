import { PassThrough } from 'stream'

import { Box, renderSync, ScrollBox, type ScrollBoxHandle, Text } from '@hermes/ink'
import React, { useRef } from 'react'
import { describe, expect, it } from 'vitest'

import { useVirtualHistory } from '../hooks/useVirtualHistory.js'

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

// Mirrors appLayout.tsx in INLINE_MODE: no AlternateScreen, so nothing bounds
// the transcript ScrollBox to the terminal height. The rendered frame is
// written to the primary buffer and whatever overflows the top lands in the
// host terminal's scrollback (dashboard xterm.js / Termux).
function InlineTranscript({ items, mountAll }: { items: readonly { key: string }[]; mountAll: boolean }) {
  const scrollRef = useRef<ScrollBoxHandle | null>(null)
  const v = useVirtualHistory(scrollRef, items, 60, { estimateHeight: () => 2, mountAll })

  return React.createElement(
    Box,
    { flexDirection: 'column', flexGrow: 1 },
    React.createElement(
      ScrollBox,
      { flexDirection: 'column', flexGrow: 1, flexShrink: 1, ref: scrollRef, stickyScroll: true },
      React.createElement(
        Box,
        { flexDirection: 'column' },
        v.topSpacer > 0 ? React.createElement(Box, { height: v.topSpacer }) : null,
        ...items
          .slice(v.start, v.end)
          .map(item =>
            React.createElement(
              Box,
              { flexDirection: 'column', key: item.key, ref: v.measureRef(item.key) },
              React.createElement(Text, null, `message ${item.key}`),
              React.createElement(Text, null, 'body')
            )
          ),
        v.bottomSpacer > 0 ? React.createElement(Box, { height: v.bottomSpacer }) : null
      )
    ),
    React.createElement(Text, null, 'STATUS')
  )
}

const renderInlineFrame = async (itemCount: number) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let frame = ''

  Object.assign(stdout, { columns: 60, isTTY: false, rows: 20 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    frame = String(chunk)
  })

  const items = Array.from({ length: itemCount }, (_, i) => ({ key: `m${i}` }))

  const instance = renderSync(React.createElement(InlineTranscript, { items, mountAll: true }), {
    exitOnCtrlC: false,
    patchConsole: false,
    stderr,
    stdin,
    stdout
  } as Parameters<typeof renderSync>[1])

  await delay(50)
  const lines = frame.split('\n')
  instance.unmount()

  return lines
}

describe('useVirtualHistory inline (primary-buffer) rendering', () => {
  it('writes the whole transcript to scrollback instead of blank spacer rows', async () => {
    // Well above the default mounted-row cap so windowing would kick in.
    const lines = await renderInlineFrame(300)
    const firstMessage = lines.findIndex(line => line.includes('message m0'))

    expect(firstMessage).toBeGreaterThanOrEqual(0)
    expect(lines.slice(0, firstMessage).every(line => line.trim() === '')).toBe(true)
    expect(firstMessage).toBeLessThan(2)

    for (const key of ['m0', 'm150', 'm299']) {
      expect(lines.some(line => line.includes(`message ${key}`))).toBe(true)
    }

    // The frame is exactly the transcript (2 rows per item) plus the status
    // row: no hidden blank rows that the terminal would push into scrollback.
    const blankRuns = lines.reduce(
      (acc, line) => {
        const run = line.replace(/\x1b\[[0-9;?]*[a-zA-Z]/g, '').trim() === '' ? acc.run + 1 : 0

        return { max: Math.max(acc.max, run), run }
      },
      { max: 0, run: 0 }
    )

    expect(blankRuns.max).toBeLessThanOrEqual(2)
  })
})
