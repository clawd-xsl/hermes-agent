/** Compatibility for SDKs that fetch ordinary recipient keys anonymously. */
export function installAuthenticatedPreKeys(client, libsignal) {
  if (typeof client.fetchPreKeyBundles !== "function") return;
  const original = client.fetchPreKeyBundles.bind(client);
  client.fetchPreKeyBundles = async function (params) {
    // Sealed-sender callers supply their own explicit authorization.
    if (params.preKeyAuth !== undefined) return original(params);
    const connection = this.connection;
    if (!connection?.fetch) throw new Error("Signal authenticated connection unavailable for recipient keys");
    const target = typeof params.destination === "string"
      ? params.destination : params.destination.getServiceIdString();
    const device = params.device && params.device !== "all" ? params.device.deviceId : "*";
    const response = await connection.fetch({
      verb: "GET",
      path: `/v2/keys/${encodeURIComponent(target)}/${device}?pq=true`,
      headers: [["Accept", "application/json"]],
      timeoutMillis: 30000,
    }, params.abortSignal ? {abortSignal: params.abortSignal} : undefined);
    if (response.status !== 200) throw new Error(`Signal authenticated recipient keys: HTTP ${response.status}`);
    const body = JSON.parse(Buffer.from(response.body).toString("utf8"));
    const {PublicKey, KEMPublicKey, PreKeyBundle} = libsignal;
    const bytes = value => Buffer.from(value, "base64");
    const identity = PublicKey.deserialize(bytes(body.identityKey));
    return body.devices.map(d => PreKeyBundle.new(
      d.registrationId, d.deviceId,
      d.preKey?.keyId ?? null, d.preKey ? PublicKey.deserialize(bytes(d.preKey.publicKey)) : null,
      d.signedPreKey.keyId, PublicKey.deserialize(bytes(d.signedPreKey.publicKey)), bytes(d.signedPreKey.signature),
      identity, d.pqPreKey.keyId, KEMPublicKey.deserialize(bytes(d.pqPreKey.publicKey)), bytes(d.pqPreKey.signature),
    ));
  };
}
