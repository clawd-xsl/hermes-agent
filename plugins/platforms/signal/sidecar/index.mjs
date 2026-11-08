#!/usr/bin/env node

/**
 * Persistent signal-ts host for the Hermes Signal adapter.
 *
 * Protocol: newline-delimited JSON over stdio. stdout carries protocol records only; diagnostics
 * go to stderr. This process owns the single authenticated Signal socket and every libsignal
 * state mutation for the lifetime of the gateway. On a lost connection it exits (status 75) and
 * the Python supervisor starts a fresh process; configuration errors exit 78 after a structured
 * ``fatal`` record so the gateway does not retry them.
 *
 * Usage: node index.mjs --sdk-path <dir|entry> --state-path <file> --cache-dir <dir>
 *                       [--expected-account <+E164>]
 */

import { createInterface } from "node:readline";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { randomUUID } from "node:crypto";
import { createRequire } from "node:module";
import { installAuthenticatedPreKeys } from "./prekeys.mjs";

const MIN_NODE = [22, 19];
const MAX_ATTACHMENTS = 1024;
const SEND_RETRY_DELAYS_MS = [750, 2000];
const EXIT_RECONNECT = 75;
const EXIT_CONFIG = 78;

class ConfigError extends Error {
  constructor(code, message) {
    super(message);
    this.name = "ConfigError";
    this.code = code;
  }
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

/** Server rate-limit hint anywhere on the error chain (libsignal RateLimitedError.retryAfterSecs). */
function rateLimitInfo(error) {
  for (let current = error, depth = 0; current && depth < 8; current = current.cause, depth += 1) {
    const text = `${current?.name ?? ""} ${current?.message ?? ""}`;
    const retry = current?.retryAfterSecs ?? current?.retryAfterSeconds;
    if (typeof retry === "number" && Number.isFinite(retry)) {
      return { retryAfterSeconds: retry };
    }
    if (/ratelimit|\b429\b|retry after/i.test(text)) {
      const match = /retry after (\d+(?:\.\d+)?)\s*second/i.exec(text);
      return { retryAfterSeconds: match ? Number(match[1]) : null };
    }
  }
  return null;
}

function errorRecord(error) {
  const record = {
    name: error instanceof Error ? error.name : "Error",
    message: describeError(error),
    code: error && typeof error === "object" && "code" in error ? String(error.code) : undefined,
  };
  const rateLimit = rateLimitInfo(error);
  if (rateLimit) {
    record.rateLimited = true;
    record.retryAfterSeconds = rateLimit.retryAfterSeconds;
    record.message = `[429] Rate limit exceeded: ${record.message}`;
  }
  return record;
}

function isUnlinkedError(error) {
  const text = describeError(error).toLowerCase();
  return (
    text.includes("devicedelinked") ||
    text.includes("unauthorized") ||
    text.includes("http 401") ||
    text.includes("http 403")
  );
}

/** Report a startup failure and exit once stdout is flushed (pipes are async on macOS). */
function fatal(code, message, retryable) {
  process.stdout.write(`${JSON.stringify({ event: "fatal", code, message, retryable })}\n`, () => {
    process.exit(retryable ? 1 : EXIT_CONFIG);
  });
  return new Promise(() => {});
}

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new ConfigError("bad_arguments", `Invalid signal-ts sidecar argument near ${key ?? "<end>"}`);
    }
    result[key.slice(2)] = value;
  }
  for (const required of ["sdk-path", "state-path", "cache-dir"]) {
    if (!result[required]) {
      throw new ConfigError("bad_arguments", `Missing --${required}`);
    }
  }
  return result;
}

function checkNodeVersion() {
  const [major, minor] = process.versions.node.split(".").map(Number);
  if (major < MIN_NODE[0] || (major === MIN_NODE[0] && minor < MIN_NODE[1])) {
    throw new ConfigError(
      "node_too_old",
      `signal-ts requires Node >= ${MIN_NODE.join(".")}; ${process.execPath} is ${process.versions.node}`,
    );
  }
}

async function resolveSdkEntry(configuredPath) {
  const candidate = path.resolve(configuredPath);
  let info;
  try {
    info = await stat(candidate);
  } catch {
    throw new ConfigError("sdk_missing", `signal-ts sdk_path does not exist: ${candidate}`);
  }
  if (info.isFile()) {
    return candidate;
  }
  let entry = "dist/index.js";
  try {
    const manifest = JSON.parse(await readFile(path.join(candidate, "package.json"), "utf8"));
    if (typeof manifest.main === "string") {
      entry = manifest.main;
    }
  } catch {
    throw new ConfigError("sdk_missing", `signal-ts sdk_path has no readable package.json: ${candidate}`);
  }
  const resolved = path.resolve(candidate, entry);
  try {
    await stat(resolved);
  } catch {
    throw new ConfigError(
      "sdk_unbuilt",
      `signal-ts SDK is not built (${resolved} missing); run: pnpm install --frozen-lockfile && pnpm build`,
    );
  }
  return resolved;
}

function normalizeAci(raw) {
  let value = String(raw ?? "").trim();
  value = value.replace(/^(?:signal:|uuid:|aci:)/i, "");
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value)
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
      ".mov": "video/quicktime",
      ".mp3": "audio/mpeg",
      ".mp4": "video/mp4",
      ".ogg": "audio/ogg",
      ".opus": "audio/ogg",
      ".pdf": "application/pdf",
      ".png": "image/png",
      ".txt": "text/plain",
      ".wav": "audio/wav",
      ".webm": "video/webm",
      ".webp": "image/webp",
      ".zip": "application/zip",
    }[extension] ?? "application/octet-stream"
  );
}

function bodyRangesFromParams(params) {
  const styleCodes = { BOLD: 1, ITALIC: 2, SPOILER: 3, STRIKETHROUGH: 4, MONOSPACE: 5 };
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
    return [{ start: Number(match[1]), length: Number(match[2]), style: styleCodes[match[3]] }];
  });
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

async function openRuntime(args) {
  checkNodeVersion();
  const sdkEntry = await resolveSdkEntry(args["sdk-path"]);
  let sdk;
  try {
    sdk = await import(pathToFileURL(sdkEntry).href);
  } catch (error) {
    throw new ConfigError("sdk_import_failed", `could not load the signal-ts SDK: ${describeError(error)}`);
  }
  const statePath = path.resolve(args["state-path"]);
  try {
    if (!(await stat(statePath)).isFile()) {
      throw new Error("not a file");
    }
  } catch {
    throw new ConfigError(
      "state_missing",
      `signal-ts state file does not exist: ${statePath} (link this device with signal-ts first)`,
    );
  }
  const cacheDir = path.resolve(args["cache-dir"]);
  await mkdir(cacheDir, { recursive: true, mode: 0o700 });

  let repository;
  try {
    repository = await sdk.FileSignalRepository.open(statePath);
  } catch (error) {
    throw new ConfigError("state_unreadable", `could not read signal-ts state ${statePath}: ${describeError(error)}`);
  }
  const accountState = await repository.getAccount();
  if (!accountState) {
    throw new ConfigError(
      "state_unlinked",
      `signal-ts state has no linked account: ${statePath} (link this device with signal-ts first)`,
    );
  }
  const account = accountState.account;
  const accountNumber = account.device.e164 ?? "";
  const expectedAccount = String(args["expected-account"] ?? "").trim();
  if (expectedAccount && accountNumber && expectedAccount !== accountNumber) {
    throw new ConfigError(
      "account_mismatch",
      `signal-ts state account ${accountNumber} does not match configured account ${expectedAccount}`,
    );
  }

  const client = new sdk.SignalTsClient({
    account,
    environment: "production",
    userAgent: accountState.userAgent ?? "Hermes signal-ts",
    receiveStories: accountState.receiveStories ?? false,
    logger,
  });
  // signal-ts's anonymous key lookup fails for recipients who restrict sealed sender access.
  // Ordinary authenticated sends must fetch keys with our own login.
  if (typeof client.fetchPreKeyBundles === "function") {
    const sdkRequire = createRequire(sdkEntry);
    const libsignal = await import(pathToFileURL(sdkRequire.resolve("@signalapp/libsignal-client")).href);
    installAuthenticatedPreKeys(client, libsignal);
  }
  return { sdk, repository, accountState, account, accountNumber, client, cacheDir };
}

function createHost({ sdk, repository, account, accountNumber, client, cacheDir }) {
  const stores = sdk.createLibsignalStores(repository);
  const localAddress = sdk.createSignalLocalAddress(account);
  const attachments = new Map();
  const inboundTasks = new Set();
  // Inbound envelopes are handled strictly in arrival order: Signal delivers a chat's messages
  // in order and the gateway's serial receive loop relies on it.
  let inboundChain = Promise.resolve();
  const state = { shuttingDown: false };

  function rememberAttachment(pointer) {
    const id = `signal-ts:${randomUUID()}`;
    attachments.set(id, pointer);
    while (attachments.size > MAX_ATTACHMENTS) {
      attachments.delete(attachments.keys().next().value);
    }
    return id;
  }

  function attachmentRecords(pointers) {
    return (pointers ?? []).map((pointer) => ({
      id: rememberAttachment(pointer),
      contentType: pointer.contentType ?? null,
      filename: pointer.fileName ?? null,
      size: pointer.size ?? null,
    }));
  }

  function mentionsFromBodyRanges(bodyRanges) {
    return (bodyRanges ?? [])
      .filter((range) => range.mentionAci && Number.isFinite(range.start))
      .map((range) => ({ start: range.start, length: range.length ?? 1, uuid: range.mentionAci }));
  }

  function quoteRecord(quote) {
    return quote
      ? { id: quote.id ?? null, text: quote.text ?? null, authorUuid: quote.authorAci ?? null }
      : undefined;
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
      attachments: attachmentRecords(message.attachments),
      mentions: mentionsFromBodyRanges(message.bodyRanges),
    };
    if (message.group?.id) {
      dataMessage.groupInfo = { groupId: message.group.id, groupName: group?.title ?? null };
    }
    const quote = quoteRecord(message.message?.quote);
    if (quote) {
      dataMessage.quote = quote;
    }
    return { ...base, dataMessage };
  }

  async function syncMessageToEnvelope(message) {
    const sync = message.syncMessage;
    if (!sync || typeof sync !== "object") {
      return null;
    }
    // An already-normalized gateway envelope passes through; the real SDK exposes protobufjs'
    // SyncMessage.Sent shape under ``sent``.
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
    const destinationIsSelf = normalizeAci(destinationServiceId) === normalizeAci(account.device.aci);
    const timestamp = sent.timestamp ?? message.timestamp ?? message.serverTimestamp ?? null;
    const sentMessage = {
      destinationNumber: sent.destinationE164 ?? (destinationIsSelf ? accountNumber || null : null),
      destination: sent.destinationE164 ?? (destinationServiceId || null),
      timestamp,
      message: data.body ?? null,
      attachments: attachmentRecords(data.attachments),
      mentions: mentionsFromBodyRanges(data.bodyRanges),
    };
    if (groupId) {
      sentMessage.groupInfo = { groupId, groupName: group?.title ?? null };
    }
    const quote = quoteRecord(data.quote);
    if (quote) {
      sentMessage.quote = quote;
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
      // Note to Self and primary-device group sends; attachment pointers stay in the sidecar.
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
      return { ...envelope, dataMessage: undefined, editMessage: { dataMessage: envelope.dataMessage } };
    }
    // Reactions, receipts, typing, calls and sync metadata do not start agent turns; Hermes
    // sends its own lifecycle reactions through explicit commands.
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
      for (const message of sdk.normalizeDecryptedIncomingMessage(decrypted)) {
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
        await incoming.ack();
      } catch (error) {
        logger.error("inbound ack failed", error);
      }
    }
  }

  function onIncoming(incoming) {
    writeRecord({ event: "transport", timestamp: Date.now() });
    const task = inboundChain.then(() => handleIncoming(incoming));
    inboundChain = task.catch(() => undefined);
    const tracked = task.finally(() => inboundTasks.delete(tracked));
    inboundTasks.add(tracked);
  }

  async function resolveTarget(raw) {
    const parsed = sdk.parseSignalRecipientTarget(raw);
    if (parsed.kind === "e164") {
      const known = await repository.getRecipientByE164(parsed.e164);
      return known?.aci ?? { kind: "e164", e164: parsed.e164 };
    }
    return raw;
  }

  async function recipientRecord(raw) {
    const parsed = sdk.parseSignalRecipientTarget(raw);
    if (parsed.kind === "e164") {
      return await repository.getRecipientByE164(parsed.e164);
    }
    if (parsed.kind === "aci") {
      return await repository.getRecipientByAci(
        typeof parsed.aci === "string" ? parsed.aci : parsed.aci.getServiceIdString(),
      );
    }
    return undefined;
  }

  async function resolvePreKeyAuth(raw) {
    const recipient = await recipientRecord(raw);
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
    // One timestamp across retries so a duplicate delivery dedupes on the recipient's side.
    const timestamp = Date.now();
    for (let attempt = 0; ; attempt += 1) {
      try {
        return await run(timestamp);
      } catch (error) {
        const delay = SEND_RETRY_DELAYS_MS[attempt];
        const text = describeError(error).toLowerCase();
        // Rate limits are paced by the Python scheduler from the server's hint; never hammer here.
        const fatalError =
          rateLimitInfo(error) !== null ||
          text.includes("connectedelsewhere") ||
          text.includes("authentication") ||
          text.includes("unauthorized") ||
          text.includes("forbidden") ||
          text.includes("identity key");
        if (delay === undefined || fatalError) {
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
        attachment: { data, contentType: mimeFromPath(filePath), fileName: path.basename(filePath) },
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
      // signal-ts does not expose group typing sends yet; typing is best-effort.
      return true;
    }
    const rawTarget = recipientParam(params);
    const destination = await resolveTarget(rawTarget);
    const preKeyAuth = await resolvePreKeyAuth(rawTarget);
    await client.sendTypingMessage({
      destination,
      typing: { timestamp: Date.now(), action: params.stop ? "stopped" : "started" },
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
      const recipient = await recipientRecord(value);
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
    const data = await sdk.downloadSignalAttachment({ pointer, fetch: sdk.signalAttachmentFetch });
    const filePath = path.join(cacheDir, `${randomUUID()}.attachment`);
    await writeFile(filePath, data, { mode: 0o600 });
    attachments.delete(id);
    return { path: filePath, contentType: pointer.contentType ?? null };
  }

  function contactRecord(recipient) {
    return { number: recipient.e164 ?? null, uuid: recipient.aci, name: recipient.name ?? null };
  }

  function listContacts() {
    const snapshot = repository.snapshot();
    return Object.values(snapshot.recipients ?? {}).map(contactRecord);
  }

  async function getContact(params) {
    const target = String(params.contactAddress ?? "").trim();
    const aci = normalizeAci(target);
    const recipient = aci ? await repository.getRecipientByAci(aci) : await repository.getRecipientByE164(target);
    return recipient ? contactRecord(recipient) : null;
  }

  async function shutdown() {
    state.shuttingDown = true;
    await Promise.allSettled([...inboundTasks]);
    await client.disconnect();
    return true;
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
          lastTransportActivityAt: client.getLastTransportActivityAt?.() ?? Date.now(),
        };
      case "shutdown":
        return await shutdown();
      default:
        throw new Error(`Unsupported signal-ts method: ${method}`);
    }
  }

  return { state, inboundTasks, onIncoming, handleRequest, shutdown };
}

function serveRequests(host) {
  const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
  input.on("line", (line) => {
    void (async () => {
      let request;
      try {
        request = JSON.parse(line);
        if (!request || typeof request !== "object" || typeof request.id !== "string") {
          throw new Error("Invalid signal-ts request");
        }
        const result = await host.handleRequest(String(request.method ?? ""), request.params ?? {});
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
  // The gateway closing our stdin means it is gone: never outlive it as a second client.
  input.on("close", () => {
    if (!host.state.shuttingDown) {
      void host.shutdown().finally(() => process.exit(0));
    }
  });
}

async function main() {
  let runtime;
  try {
    runtime = await openRuntime(parseArgs(process.argv.slice(2)));
  } catch (error) {
    if (error instanceof ConfigError) {
      await fatal(error.code, error.message, false);
    }
    await fatal("startup_failed", `signal-ts startup failed: ${describeError(error)}`, true);
  }
  const host = createHost(runtime);
  const { client } = runtime;
  client.on("incoming", host.onIncoming);
  client.on("disconnected", (error) => {
    if (host.state.shuttingDown) {
      return;
    }
    logger.error("Signal connection lost", error);
    // The Python supervisor owns bounded backoff and starts a fresh process. A clean process
    // boundary guarantees no zombie client after ConnectedElsewhere or a dead keepalive.
    process.exitCode = EXIT_RECONNECT;
    void Promise.race([
      Promise.allSettled([...host.inboundTasks]),
      new Promise((resolve) => setTimeout(resolve, 2000)),
    ]).finally(() => process.exit(EXIT_RECONNECT));
  });

  try {
    await client.connect();
  } catch (error) {
    if (isUnlinkedError(error)) {
      await fatal("unlinked", `Signal rejected this linked device (relink it): ${describeError(error)}`, false);
    }
    await fatal("connect_failed", `Signal connection failed: ${describeError(error)}`, true);
  }
  writeRecord({
    event: "ready",
    account: runtime.accountNumber || null,
    aci: runtime.account.device.aci,
    deviceId: runtime.account.device.deviceId,
    timestamp: Date.now(),
  });
  serveRequests(host);
  if (typeof client.startIncomingDelivery === "function") {
    await client.startIncomingDelivery();
  }

  const stop = () => {
    host.state.shuttingDown = true;
    void client.disconnect().finally(() => process.exit(0));
  };
  process.on("SIGTERM", stop);
  process.on("SIGINT", stop);
}

process.on("uncaughtException", (error) => {
  logger.error("uncaught exception", error);
  process.exit(70);
});

process.on("unhandledRejection", (error) => {
  logger.error("unhandled rejection", error);
  process.exit(70);
});

await main();
