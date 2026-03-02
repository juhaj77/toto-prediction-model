// ─── MultiHeadAttention.js ───────────────────────────────────────────────────
// Custom TF.js layer implementing Multi-Head Self-Attention.
//
// TF.js does not ship MultiHeadAttention in its layers API, so it must be
// implemented and registered manually BEFORE tf.loadLayersModel() is called.
//
// Usage (App.jsx or any file that loads a model using this layer):
//   import { registerMultiHeadAttention } from './MultiHeadAttention.js';
//   registerMultiHeadAttention(); // call once, before tf.loadLayersModel()
//
// The ravimalli-race.js training script imports this via:
//   const { MultiHeadAttention } = require('./MultiHeadAttention.js')
// That file stays CommonJS — this file is the ES module browser version.
// ─────────────────────────────────────────────────────────────────────────────

import * as tf from '@tensorflow/tfjs';

class MultiHeadAttention extends tf.layers.Layer {
    static className = "MultiHeadAttention";

    constructor(args) {
        super(args);
        const { numHeads, embedDim, useBias, dropout, causal } = args;

        if (embedDim % numHeads !== 0) {
            throw Error(`MultiHeadAttention: embedDim (${embedDim}) is not divisible by numHeads (${numHeads})`);
        }

        this.numHeads = numHeads;
        this.embedDim = embedDim;
        this.useBias = useBias ?? true;
        this.dropout = dropout ?? 0.0;
        this.causal = causal ?? false;

        // Projektio-kerrokset (Dense)
        this.queryProjection = tf.layers.dense({ useBias: this.useBias, units: embedDim });
        this.keyProjection = tf.layers.dense({ useBias: this.useBias, units: embedDim });
        this.valueProjection = tf.layers.dense({ useBias: this.useBias, units: embedDim });
        this.outputProjection = tf.layers.dense({ useBias: this.useBias, units: embedDim });
        this.dropoutLayer = (this.dropout && this.dropout > 0)
            ? tf.layers.dropout({ rate: this.dropout })
            : null;
    }

    build(inputShape) {
        // Alustetaan alikerrokset
        const shape = Array.isArray(inputShape) && Array.isArray(inputShape[0]) ? inputShape[0] : inputShape;

        this.queryProjection.build(shape);
        this.keyProjection.build(shape);
        this.valueProjection.build(shape);
        this.outputProjection.build(shape);

        // Alikerrokset hoitavat omien painojensa rekisteröinnin automaattisesti
        // eikä niitä tarvitse koota käsin tähän kerrokseen.
        super.build(inputShape);
    }

    call(inputs, kwargs) {
        // Wrap the whole call in a tidy and keep the final tensor so that
        // upstream tf.layers (which also use tidy) won't accidentally dispose it.
        return tf.tidy(() => {
            // Support signatures:
            //  - inputs: x → q=k=v=x (no mask)
            //  - inputs: [q, k, v]
            //  - inputs: [q, k, v, staticInput] → derive key mask from staticInput[:, :, 0] > 0
            let query, key, value, staticInputForMask = null;
            if (Array.isArray(inputs)) {
                if (inputs.length === 1) {
                    query = inputs[0]; key = inputs[0]; value = inputs[0];
                } else if (inputs.length >= 3) {
                    [query, key, value] = inputs;
                    if (inputs.length >= 4) staticInputForMask = inputs[3];
                } else {
                    // Fallback
                    query = inputs[0]; key = inputs[0]; value = inputs[0];
                }
            } else {
                query = inputs; key = inputs; value = inputs;
            }

            key = key || query;
            value = value || query;

            const batchSize = query.shape[0];

            // 1. Projektiot
            let q = this.queryProjection.apply(query);
            let k = this.keyProjection.apply(key);
            let v = this.valueProjection.apply(value);

            // 2. Split heads: [batch, seq, dims] -> [batch, heads, seq, dims/heads]
            const splitHeads = (x) => {
                const reshaped = x.reshape([batchSize, -1, this.numHeads, this.embedDim / this.numHeads]);
                return reshaped.transpose([0, 2, 1, 3]);
            };

            q = splitHeads(q);
            k = splitHeads(k);
            v = splitHeads(v);

            // 3. Scaled Dot-Product Attention
            const depth = k.shape[k.shape.length - 1];
            let logits = q.matMul(k, false, true).div(Math.sqrt(depth));

            // Optional causal mask
            if (this.causal) {
                const seqLen = logits.shape[logits.shape.length - 1];
                const mask = tf.linalg.bandPart(tf.ones([seqLen, seqLen]), -1, 0).sub(1).mul(1e9);
                logits = logits.add(mask);
            }

            // Optional key padding mask derived from static input's first feature (start number > 0)
            if (staticInputForMask) {
                // staticInput shape: [batch, seq, features]
                const firstFeat = tf.slice(staticInputForMask, [0, 0, 0], [-1, -1, 1]).squeeze([-1]); // [batch, seq]
                const keyMask2d = firstFeat.greater(tf.scalar(0)).cast('float32');
                // Expand to [batch, 1, 1, seq_k] so it broadcasts over heads and query length
                const keyMask4d = keyMask2d.expandDims(1).expandDims(1);
                const negInf = tf.scalar(-1e9);
                // logits shape: [batch, heads, seq_q, seq_k]
                const one = tf.scalar(1.0);
                const invMask = one.sub(keyMask4d);
                logits = logits.add(invMask.mul(negInf));
            }

            let weights = tf.softmax(logits, -1);
            if (this.dropoutLayer) {
                // Keras will propagate `training` down the call stack.
                weights = this.dropoutLayer.apply(weights, { training: kwargs?.training });
            }
            let attentionOut = weights.matMul(v);

            // 4. Concat heads: [batch, heads, seq, dims/heads] -> [batch, seq, dims]
            attentionOut = attentionOut.transpose([0, 2, 1, 3]).reshape([batchSize, -1, this.embedDim]);

            // 5. Output projektio
            const out = this.outputProjection.apply(attentionOut);
            return tf.keep(out);
        });
    }

    computeOutputShape(inputShape) {
        return Array.isArray(inputShape) && Array.isArray(inputShape[0]) ? inputShape[0] : inputShape;
    }

    getConfig() {
        const config = {
            numHeads: this.numHeads,
            embedDim: this.embedDim,
            useBias: this.useBias,
            dropout: this.dropout,
            causal: this.causal
        };
        const baseConfig = super.getConfig();
        return Object.assign({}, baseConfig, config);
    }
}

// Rekisteröidään kerros globaalisti
//tf.serialization.registerClass(MultiHeadAttention);

// ─── Registration helper ──────────────────────────────────────────────────────
// Call this once before tf.loadLayersModel().
// Safe to call multiple times (tf.serialization ignores duplicate registrations).

export function registerMultiHeadAttention() {
    tf.serialization.registerClass(MultiHeadAttention);
}
