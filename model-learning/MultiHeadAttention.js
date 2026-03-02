const tf = require('@tensorflow/tfjs');

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
            let [query, key, value] = Array.isArray(inputs) ? inputs : [inputs, inputs, inputs];
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

            if (this.causal) {
                const seqLen = logits.shape[logits.shape.length - 1];
                const mask = tf.linalg.bandPart(tf.ones([seqLen, seqLen]), -1, 0).sub(1).mul(1e9);
                logits = logits.add(mask);
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
tf.serialization.registerClass(MultiHeadAttention);

module.exports = { MultiHeadAttention };