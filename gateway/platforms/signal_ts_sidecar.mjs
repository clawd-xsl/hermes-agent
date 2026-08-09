#!/usr/bin/env node

/**
 * Persistent signal-ts host for Hermes.
 *
 * The protocol is newline-delimited JSON over stdio.  stdout is protocol-only;
 * diagnostics go to stderr.  This process deliberately owns the single Signal
 * socket and all libsignal mutations for the lifetime of the gateway.
 */

import { createInterface } from "node:readline";
import { mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { randomUUID } from "node:crypto";

const MAX_ATTACHMENTS = 1024;
const SEND_RETRY_DELAYS_MS = [750, 2000];

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(`Invalid signal-ts sidecar argument near ${key ?? "<end>"}`);
    }
    result[key.slice(2)] = value;
  }
  return result;
}

function writeRecord(record) {
  process.stdout.write(`${JSON.stringify(record)}\n`);
}

function describeError(error) {
  if (error instanceof Error) {
    const cause = error.cause ? `; cause=${describeError(error.cause)}` : "";
    return `${error.name}: ${error.message}${cause}`;
  }
  return String(error);
}

function errorRecord(error) {
  return {
    name: error instanceof Error ? error.name : "Error",
    message: describeError(error),
    code:
      error && typeof error === "object" && "code" in error
        ? String(error.code)
        : undefined,
  };
}

async function resolveSdkEntry(configuredPath) {
  const candidate = path.resolve(configuredPath);
  const info = await stat(candidate);
  if (info.isFile()) {
    return candidate;
  }
  if (!info.isDirectory()) {
    throw new Error(`signal-ts sdk_path is not a file or directory: ${candidate}`);
  }
  const manifestPath = path.join(candidate, "package.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  const entry = typeof manifest.main === "string" ? manifest.main : "dist/index.js";
  return path.resolve(candidate, entry);
}

function normalizeAci(raw) {
  let value = String(raw ?? "").trim();
  value = value.replace(/^(?:signal:|uuid:|aci:)/i, "");
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
    value,
  )
    ? value.toLowerCase()
    : undefined;
}

function parseGroupId(params) {
  const groupId = String(params.groupId ?? "").trim();
  return groupId || undefined;
}

function recipientParam(params) {
  const values = Array.isArray(params.recipient) ? params.recipient : [];
  const value = String(values[0] ?? params.recipient ?? "").trim();
  if (!value) {
    throw new Error("Signal recipient is required");
  }
  return value;
}

function mimeFromPath(filePath) {
  const extension = path.extname(filePath).toLowerCase();
  return (
    {
      ".aac": "audio/aac",
      ".gif": "image/gif",
      ".jpeg": "image/jpeg",
      ".jpg": "image/jpeg",
      ".m4a": "audio/mp4",
      ".mp3": "audio/mpeg",
      ".mp4": "video/mp4",
      ".ogg": "audio/ogg",
      ".pdf": "application/pdf",
      ".png": "image/png",
      ".wav": "audio/wav",
      ".webp": "image/webp",
      ".zip": "application/zip",
    }[extension] ?? "application/octet-stream"
  );
}

function bodyRangesFromParams(params) {
  const styleCodes = {
    BOLD: 1,
    ITALIC: 2,
    SPOILER: 3,
    STRIKETHROUGH: 4,
    MONOSPACE: 5,
  };
  const values = [];
  if (typeof params.textStyle === "string") {
    values.push(params.textStyle);
  }
  if (Array.isArray(params.textStyles)) {
    values.push(...params.textStyles);
  }
  return values.flatMap((raw) => {
    const match = /^(\d+):(\d+):([A-Z]+)$/.exec(String(raw));
    if (!match || !(match[3] in styleCodes)) {
      return [];
    }
    return [{
      start: Number(match[1]),
      length: Number(match[2]),
      style: styleCodes[match[3]],
    }];
  });
}

const args = parseArgs(process.argv.slice(2));
for (const required of ["sdk-path", "state-path", "cache-dir"]) {
  if (!args[required]) {
    throw new Error(`Missing --${required}`);
  }
}

const sdkEntry = await resolveSdkEntry(args["sdk-path"]);
const sdk = await import(pathToFileURL(sdkEntry).href);
const statePath = path.resolve(args["state-path"]);
const cacheDir = path.resolve(args["cache-dir"]);
await mkdir(cacheDir, { recursive: true, mode: 0o700 });

const repository = await sdk.FileSignalRepository.open(statePath);
const accountState = await repository.getAccount();
if (!accountState) {
  throw new Error(`signal-ts state is missing account data: ${statePath}`);
}
const account = accountState.account;
const accountNumber = account.device.e164 ?? "";
const expectedAccount = String(args["expected-account"] ?? "").trim();
if (expectedAccount && accountNumber && expectedAccount !== accountNumber) {
  throw new Error(
    `signal-ts state account ${accountNumber} does not match configured account ${expectedAccount}`,
  );
}

const logger = {
  debug(message) {
    process.stderr.write(`[debug] ${message}\n`);
  },
  info(message) {
    process.stderr.write(`[info] ${message}\n`);
  },
  warn(message) {
    process.stderr.write(`[warn] ${message}\n`);
  },
  error(message, error) {
    process.stderr.write(`[error] ${message}${error ? `: ${describeError(error)}` : ""}\n`);
  },
};

const client = new sdk.SignalTsClient({
  account,
  environment: "production",
  userAgent: accountState.userAgent ?? "Hermes signal-ts",
  receiveStories: accountState.receiveStories ?? false,
  logger,
});
const stores = sdk.createLibsignalStores(repository);
const localAddress = sdk.createSignalLocalAddress(account);
const attachments = new Map();
const inboundTasks = new Set();
let shuttingDown = false;

function rememberAttachment(pointer) {
  const id = `signal-ts:${randomUUID()}`;
  attachments.set(id, pointer);
  while (attachments.size > MAX_ATTACHMENTS) {
    attachments.delete(attachments.keys().next().value);
  }
  return id;
}

function mentionsFromBodyRanges(bodyRanges) {
  return (bodyRanges ?? [])
    .filter((range) => range.mentionAci && Number.isFinite(range.start))
    .map((range) => ({
      start: range.start,
      length: range.length ?? 1,
      uuid: range.mentionAci,
    }));
}

async function sourceFields(message) {
  const sourceUuid = message.sender?.serviceId ?? null;
  const recipient = sourceUuid ? await repository.getRecipientByAci(sourceUuid) : undefined;
  return {
    sourceUuid,
    sourceNumber: recipient?.e164 ?? null,
    sourceName: recipient?.name ?? null,
    timestamp: message.timestamp ?? message.serverTimestamp ?? null,
  };
}

async function dataMessageToEnvelope(message) {
  const base = await sourceFields(message);
  const group = message.group?.id ? await repository.getGroup(message.group.id) : undefined;
  const dataMessage = {
    timestamp: message.timestamp ?? message.serverTimestamp ?? null,
    message: message.body ?? null,
    attachments: (message.attachments ?? []).map((pointer) => ({
      id: rememberAttachment(pointer),
      contentType: pointer.contentType ?? null,
      filename: pointer.fileName ?? null,
      size: pointer.size ?? null,
    })),
    mentions: mentionsFromBodyRanges(message.bodyRanges),
  };
  if (message.group?.id) {
    dataMessage.groupInfo = {
      groupId: message.group.id,
      groupName: group?.title ?? null,
    };
  }
  if (message.message?.quote) {
    dataMessage.quote = {
      id: message.message.quote.id ?? null,
      text: message.message.quote.text ?? null,
      authorUuid: message.message.quote.authorAci ?? null,
    };
  }
  return { ...base, dataMessage };
}

async function syncMessageToEnvelope(message) {
  const sync = message.syncMessage;
  if (!sync || typeof sync !== "object") {
    return null;
  }
  // Keep compatibility with an already-normalized gateway fixture, but the
  // real SDK exposes protobufjs' SyncMessage.Sent shape under ``sent``.
  if (sync.sentMessage && typeof sync.sentMessage === "object") {
    return {
      sourceUuid: account.device.aci,
      sourceNumber: accountNumber || null,
      timestamp: message.timestamp ?? message.serverTimestamp ?? null,
      syncMessage: { sentMessage: sync.sentMessage },
    };
  }
  const sent = sync.sent;
  const data = sent?.message ?? sent?.editMessage?.dataMessage;
  if (!sent || !data) {
    return null;
  }

  let groupId;
  if (data.groupV2?.masterKey && sdk.signalGroupIdFromMasterKey) {
    groupId = sdk.signalGroupIdFromMasterKey(data.groupV2.masterKey);
  }
  const group = groupId ? await repository.getGroup(groupId) : undefined;
  const destinationServiceId = String(sent.destinationServiceId ?? "").trim();
  const destinationIsSelf =
    normalizeAci(destinationServiceId) === normalizeAci(account.device.aci);
  const timestamp = sent.timestamp ?? message.timestamp ?? message.serverTimestamp ?? null;
  const sentMessage = {
    destinationNumber:
      sent.destinationE164 ?? (destinationIsSelf ? accountNumber || null : null),
    destination: sent.destinationE164 ?? (destinationServiceId || null),
    timestamp,
    message: data.body ?? null,
    attachments: (data.attachments ?? []).map((pointer) => ({
      id: rememberAttachment(pointer),
      contentType: pointer.contentType ?? null,
      filename: pointer.fileName ?? null,
      size: pointer.size ?? null,
    })),
    mentions: mentionsFromBodyRanges(data.bodyRanges),
  };
  if (groupId) {
    sentMessage.groupInfo = {
      groupId,
      groupName: group?.title ?? null,
    };
  }
  if (data.quote) {
    sentMessage.quote = {
      id: data.quote.id ?? null,
      text: data.quote.text ?? null,
      authorUuid: data.quote.authorAci ?? null,
    };
  }
  return {
    sourceUuid: account.device.aci,
    sourceNumber: accountNumber || null,
    timestamp,
    syncMessage: { sentMessage },
  };
}

async function toHermesEnvelope(message) {
  if (message.kind === "sync") {
    // Convert protobufjs' nested SyncMessage.Sent/DataMessage shape into the
    // stable gateway envelope used for Note to Self and primary-device group
    // sends. This also keeps attachment pointers inside the sidecar.
    return await syncMessageToEnvelope(message);
  }
  if (message.kind === "data") {
    return await dataMessageToEnvelope(message);
  }
  if (message.kind === "edit" && message.message) {
    const promoted = {
      ...message,
      kind: "data",
      body: message.message.body,
      attachments: message.message.attachments ?? [],
      bodyRanges: message.message.bodyRanges ?? [],
    };
    const envelope = await dataMessageToEnvelope(promoted);
    return {
      ...envelope,
      dataMessage: undefined,
      editMessage: { dataMessage: envelope.dataMessage },
    };
  }
  // Reactions, receipts, typing, calls and sync metadata do not create agent
  // turns. Hermes sends its own lifecycle reactions through explicit commands.
  return null;
}

async function handleIncoming(incoming) {
  try {
    const decrypted = await client.decryptIncoming({
      envelope: incoming.envelope,
      localAddress,
      sealedSender: {
        localAci: account.device.aci,
        localDeviceId: account.device.deviceId,
        localE164: account.device.e164 ?? null,
      },
      stores,
    });
    const messages = sdk.normalizeDecryptedIncomingMessage(decrypted);
    for (const message of messages) {
      if (message.kind === "decryption-error") {
        const serviceId = message.sender?.serviceId;
        const deviceId = message.decryptionError?.deviceId ?? message.sender?.deviceId;
        if (serviceId && deviceId !== undefined) {
          await client.archiveSessionForPeer({ serviceId, deviceId, stores });
        }
        continue;
      }
      const envelope = await toHermesEnvelope(message);
      if (envelope) {
        writeRecord({ event: "envelope", envelope });
      }
    }
  } catch (error) {
    if (error instanceof sdk.SignalTsDecryptionError && error.retryReceipt) {
      try {
        await client.sendRetryReceiptMessage({
          destination: error.retryReceipt.recipientServiceId,
          retry: error.retryReceipt,
          stores,
        });
      } catch (retryError) {
        logger.error("inbound retry receipt failed", retryError);
      }
    }
    logger.error("inbound decrypt failed", error);
  } finally {
    try {
      incoming.ack();
    } catch (error) {
      logger.error("inbound ack failed", error);
    }
  }
}

client.on("incoming", (incoming) => {
  writeRecord({ event: "transport", timestamp: Date.now() });
  const task = handleIncoming(incoming).finally(() => inboundTasks.delete(task));
  inboundTasks.add(task);
});

client.on("disconnected", (error) => {
  if (shuttingDown) {
    return;
  }
  logger.error("Signal connection lost", error);
  // The Python supervisor owns bounded backoff and starts a fresh process.  A
  // clean process boundary also guarantees there is never a zombie client in
  // the active-client slot after ConnectedElsewhere or a dead keepalive.
  process.exitCode = 75;
  void Promise.race([
    Promise.allSettled(inboundTasks),
    new Promise((resolve) => setTimeout(resolve, 2000)),
  ]).finally(() => process.exit(75));
});

async function resolveTarget(raw) {
  const parsed = sdk.parseSignalRecipientTarget(raw);
  if (parsed.kind === "e164") {
    const known = await repository.getRecipientByE164(parsed.e164);
    return known?.aci ?? { kind: "e164", e164: parsed.e164 };
  }
  return raw;
}

async function resolvePreKeyAuth(raw) {
  const parsed = sdk.parseSignalRecipientTarget(raw);
  const recipient =
    parsed.kind === "e164"
      ? await repository.getRecipientByE164(parsed.e164)
      : parsed.kind === "aci"
        ? await repository.getRecipientByAci(
            typeof parsed.aci === "string" ? parsed.aci : parsed.aci.getServiceIdString(),
          )
        : undefined;
  if (recipient?.accessKey) {
    return sdk.preKeyAuthFromBase64(recipient.accessKey);
  }
  if (recipient?.profileKey) {
    const accessKey = sdk.deriveAccessKeyBase64FromProfileKeyBase64(recipient.profileKey);
    await repository.setRecipient({ ...recipient, accessKey });
    return sdk.preKeyAuthFromBase64(accessKey);
  }
  return undefined;
}

async function resolveGroup(groupId) {
  const group = await repository.getGroup(groupId);
  if (!group) {
    throw new Error(`signal-ts state is missing group ${groupId}`);
  }
  if (!group.members?.length) {
    throw new Error(`signal-ts state is missing members for group ${groupId}`);
  }
  return group;
}

function groupParams(group) {
  return {
    members: group.members,
    group: {
      masterKey: sdk.base64ToBytes(group.masterKey),
      distributionId: group.distributionId,
      ...(group.revision !== undefined ? { revision: group.revision } : {}),
    },
  };
}

async function retrySend(run) {
  const timestamp = Date.now();
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await run(timestamp);
    } catch (error) {
      const delay = SEND_RETRY_DELAYS_MS[attempt];
      const text = describeError(error).toLowerCase();
      const fatal =
        text.includes("connectedelsewhere") ||
        text.includes("authentication") ||
        text.includes("unauthorized") ||
        text.includes("forbidden") ||
        text.includes("identity key");
      if (delay === undefined || fatal) {
        throw error;
      }
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
}

async function uploadAttachments(paths) {
  const uploaded = [];
  for (const [index, filePath] of paths.entries()) {
    const data = new Uint8Array(await readFile(filePath));
    const result = await client.uploadAttachment({
      traceId: `hermes-attachment-${Date.now()}-${index}`,
      attachment: {
        data,
        contentType: mimeFromPath(filePath),
        fileName: path.basename(filePath),
      },
      fetch: sdk.signalAttachmentFetch,
    });
    uploaded.push(result.pointer);
  }
  return uploaded;
}

async function sendMessage(params) {
  const groupId = parseGroupId(params);
  const body = String(params.message ?? "");
  const attachmentPaths = Array.isArray(params.attachments)
    ? params.attachments.map((value) => path.resolve(String(value)))
    : [];
  const uploaded = await uploadAttachments(attachmentPaths);
  const bodyRanges = bodyRangesFromParams(params);
  const result = await retrySend(async (timestamp) => {
    if (groupId) {
      const group = await resolveGroup(groupId);
      return await client.sendGroupMessage({
        traceId: `hermes-group-message-${timestamp}`,
        timestamp,
        ...groupParams(group),
        body,
        ...(bodyRanges.length ? { bodyRanges } : {}),
        ...(uploaded.length ? { attachments: uploaded } : {}),
        stores,
      });
    }
    const rawTarget = recipientParam(params);
    const destination = await resolveTarget(rawTarget);
    const preKeyAuth = await resolvePreKeyAuth(rawTarget);
    return await client.sendMessage({
      traceId: `hermes-message-${timestamp}`,
      timestamp,
      destination,
      body,
      ...(bodyRanges.length ? { bodyRanges } : {}),
      ...(uploaded.length ? { attachments: uploaded } : {}),
      stores,
      ...(preKeyAuth ? { preKeyAuth } : {}),
    });
  });
  return { timestamp: result.timestamp, results: [{ type: "SUCCESS" }] };
}

async function sendTyping(params) {
  if (parseGroupId(params)) {
    // signal-ts does not yet expose group typing sends. This is intentionally
    // best-effort and never delays the actual reply.
    return true;
  }
  const rawTarget = recipientParam(params);
  const destination = await resolveTarget(rawTarget);
  const preKeyAuth = await resolvePreKeyAuth(rawTarget);
  await client.sendTypingMessage({
    destination,
    typing: {
      timestamp: Date.now(),
      action: params.stop ? "stopped" : "started",
    },
    stores,
    ...(preKeyAuth ? { preKeyAuth } : {}),
  });
  return true;
}

async function resolveReactionAuthor(params, rawTarget) {
  for (const candidate of [params.targetAuthor, rawTarget]) {
    const aci = normalizeAci(candidate);
    if (aci) {
      return aci;
    }
    const value = String(candidate ?? "").trim();
    if (!value) {
      continue;
    }
    const parsed = sdk.parseSignalRecipientTarget(value);
    const recipient =
      parsed.kind === "e164"
        ? await repository.getRecipientByE164(parsed.e164)
        : parsed.kind === "aci"
          ? await repository.getRecipientByAci(
              typeof parsed.aci === "string" ? parsed.aci : parsed.aci.getServiceIdString(),
            )
          : undefined;
    if (recipient?.aci) {
      return recipient.aci;
    }
  }
  throw new Error("Signal reaction requires a known target author ACI");
}

async function sendReaction(params) {
  const groupId = parseGroupId(params);
  const rawTarget = groupId ? String(params.targetAuthor ?? "") : recipientParam(params);
  const authorAci = await resolveReactionAuthor(params, rawTarget);
  const reaction = {
    emoji: String(params.emoji ?? ""),
    remove: Boolean(params.remove),
    targetAuthorAci: authorAci,
    targetSentTimestamp: Number(params.targetTimestamp),
  };
  const result = await retrySend(async (timestamp) => {
    if (groupId) {
      const group = await resolveGroup(groupId);
      return await client.sendGroupReactionMessage({
        traceId: `hermes-group-reaction-${timestamp}`,
        timestamp,
        ...groupParams(group),
        reaction,
        stores,
      });
    }
    const destination = await resolveTarget(rawTarget);
    const preKeyAuth = await resolvePreKeyAuth(rawTarget);
    return await client.sendReactionMessage({
      traceId: `hermes-reaction-${timestamp}`,
      timestamp,
      destination,
      reaction,
      stores,
      ...(preKeyAuth ? { preKeyAuth } : {}),
    });
  });
  return { timestamp: result.timestamp };
}

async function fetchAttachment(params) {
  const id = String(params.id ?? "");
  const pointer = attachments.get(id);
  if (!pointer) {
    return null;
  }
  const data = await sdk.downloadSignalAttachment({
    pointer,
    fetch: sdk.signalAttachmentFetch,
  });
  const filePath = path.join(cacheDir, `${randomUUID()}.attachment`);
  await writeFile(filePath, data, { mode: 0o600 });
  attachments.delete(id);
  return { path: filePath, contentType: pointer.contentType ?? null };
}

function listContacts() {
  const snapshot = repository.snapshot();
  return Object.values(snapshot.recipients ?? {}).map((recipient) => ({
    number: recipient.e164 ?? null,
    uuid: recipient.aci,
    name: recipient.name ?? null,
  }));
}

async function getContact(params) {
  const target = String(params.contactAddress ?? "").trim();
  const aci = normalizeAci(target);
  const recipient = aci
    ? await repository.getRecipientByAci(aci)
    : await repository.getRecipientByE164(target);
  return recipient
    ? { number: recipient.e164 ?? null, uuid: recipient.aci, name: recipient.name ?? null }
    : null;
}

async function handleRequest(method, params) {
  switch (method) {
    case "send":
      return await sendMessage(params);
    case "sendTyping":
      return await sendTyping(params);
    case "sendReaction":
      return await sendReaction(params);
    case "getAttachment":
      return await fetchAttachment(params);
    case "listContacts":
      return listContacts();
    case "getContact":
      return await getContact(params);
    case "ping":
      return {
        connected: true,
        lastTransportActivityAt: client.getLastTransportActivityAt() ?? Date.now(),
      };
    case "shutdown":
      shuttingDown = true;
      await Promise.allSettled(inboundTasks);
      await client.disconnect();
      return true;
    default:
      throw new Error(`Unsupported signal-ts method: ${method}`);
  }
}

await client.connect();
writeRecord({
  event: "ready",
  account: accountNumber || null,
  aci: account.device.aci,
  deviceId: account.device.deviceId,
  timestamp: Date.now(),
});

const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on("line", (line) => {
  void (async () => {
    let request;
    try {
      request = JSON.parse(line);
      if (!request || typeof request !== "object" || typeof request.id !== "string") {
        throw new Error("Invalid signal-ts request");
      }
      const result = await handleRequest(String(request.method ?? ""), request.params ?? {});
      writeRecord({ id: request.id, ok: true, result });
      if (request.method === "shutdown") {
        input.close();
        setImmediate(() => process.exit(0));
      }
    } catch (error) {
      if (request?.id) {
        writeRecord({ id: request.id, ok: false, error: errorRecord(error) });
      } else {
        logger.error("invalid request", error);
      }
    }
  })();
});

process.on("SIGTERM", () => {
  shuttingDown = true;
  void client.disconnect().finally(() => process.exit(0));
});

process.on("SIGINT", () => {
  shuttingDown = true;
  void client.disconnect().finally(() => process.exit(0));
});

process.on("uncaughtException", (error) => {
  logger.error("uncaught exception", error);
  process.exit(70);
});

process.on("unhandledRejection", (error) => {
  logger.error("unhandled rejection", error);
  process.exit(70);
});
