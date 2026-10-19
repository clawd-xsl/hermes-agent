import test from "node:test";
import assert from "node:assert/strict";
import {installAuthenticatedPreKeys} from "../../gateway/platforms/signal_ts_prekeys.mjs";

const key = Buffer.from("public-key").toString("base64");
const responseBody = {
  identityKey: key,
  devices: [1, 2, 4].map(deviceId => ({
    deviceId, registrationId: deviceId + 100,
    signedPreKey: {keyId: 10, publicKey: key, signature: key},
    pqPreKey: {keyId: 11, publicKey: key, signature: key},
  })),
};
const libsignal = {
  PublicKey: {deserialize: bytes => bytes},
  KEMPublicKey: {deserialize: bytes => bytes},
  PreKeyBundle: {new: (...args) => args},
};

test("ordinary sends fetch all recipient devices with the live authenticated connection", async () => {
  let request;
  const client = {
    fetchPreKeyBundles: () => {throw new Error("anonymous lookup must not run");},
    connection: {fetch: async r => {
      request = r;
      return {status: 200, body: Buffer.from(JSON.stringify(responseBody))};
    }},
  };
  installAuthenticatedPreKeys(client, libsignal);
  const bundles = await client.fetchPreKeyBundles({destination: {getServiceIdString: () => "recipient"}});
  assert.equal(request.path, "/v2/keys/recipient/*?pq=true");
  assert.deepEqual(bundles.map(b => b[1]), [1, 2, 4]);
  assert.equal(bundles[0][2], null); // A depleted one-time key is optional.
  assert.equal(bundles[0][3], null);
});

test("explicit sealed-sender authorization remains with the SDK", async () => {
  const params = {destination: "recipient", preKeyAuth: {kind: "access-key"}};
  const client = {fetchPreKeyBundles: async p => {assert.equal(p, params); return "sealed";}};
  installAuthenticatedPreKeys(client, libsignal);
  assert.equal(await client.fetchPreKeyBundles(params), "sealed");
});

test("authentication errors are surfaced without retrying anonymously", async () => {
  const client = {
    fetchPreKeyBundles: () => {throw new Error("anonymous lookup must not run");},
    connection: {fetch: async () => ({status: 401})},
  };
  installAuthenticatedPreKeys(client, libsignal);
  await assert.rejects(client.fetchPreKeyBundles({destination: "recipient"}), /HTTP 401/);
});
