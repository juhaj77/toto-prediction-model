const tf = require('@tensorflow/tfjs');

class MultiHeadAttention extends tf.layers.Layer {
    static className = "MultiHeadAttention";

    constructor(args) {
        super(args);
        const { numHeads, embedDim, useBias, dropout, causal } = args;
        this.numHeads = numHeads;
        this.embedDim = embedDim; // output model dimension (must equal input last-dim in our use)
        this.useBias = useBias ?? true;
        this.dropout = dropout ?? 0.0; // kept for config compatibility; no inference-time stochasticity
        this.causal = causal ?? false;
    }

    build(inputShape) {
        const shape = Array.isArray(inputShape) && Array.isArray(inputShape[0]) ? inputShape[0] : inputShape;
        const inDim = shape[shape.length - 1];
        if (this.embedDim == null) this.embedDim = inDim;
        if (this.embedDim % this.numHeads !== 0) {
            throw Error(`MultiHeadAttention: embedDim (${this.embedDim}) is not divisible by numHeads (${this.numHeads})`);
        }
        // Explicit projections via weights to avoid Dense sublayer registration issues.
        this.qKernel = this.addWeight('q_kernel', [inDim, this.embedDim], 'float32', tf.initializers.glorotUniform());
        this.kKernel = this.addWeight('k_kernel', [inDim, this.embedDim], 'float32', tf.initializers.glorotUniform());
        this.vKernel = this.addWeight('v_kernel', [inDim, this.embedDim], 'float32', tf.initializers.glorotUniform());
        this.oKernel = this.addWeight('o_kernel', [this.embedDim, this.embedDim], 'float32', tf.initializers.glorotUniform());
        if (this.useBias) {
            this.qBias = this.addWeight('q_bias', [this.embedDim], 'float32', tf.initializers.zeros());
            this.kBias = this.addWeight('k_bias', [this.embedDim], 'float32', tf.initializers.zeros());
            this.vBias = this.addWeight('v_bias', [this.embedDim], 'float32', tf.initializers.zeros());
            this.oBias = this.addWeight('o_bias', [this.embedDim], 'float32', tf.initializers.zeros());
        }
        super.build(inputShape);
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            let [query, key, value] = Array.isArray(inputs) ? inputs : [inputs, inputs, inputs];
            key = key || query;
            value = value || query;

            const batchSize = query.shape[0];
            const seqLen = query.shape[1];
            const inDim = query.shape[2];
            const depth = this.embedDim / this.numHeads;

            const project = (x, kernel, bias) => {
                const x2 = x.reshape([-1, inDim]); // [b*seq, inDim]
                let y2 = x2.matMul(kernel.read()); // [b*seq, embedDim]
                if (this.useBias) y2 = y2.add(bias.read());
                return y2.reshape([batchSize, seqLen, this.embedDim]);
            };

            // Linear projections
            let q = project(query, this.qKernel, this.qBias);
            let k = project(key,   this.kKernel, this.kBias);
            let v = project(value, this.vKernel, this.vBias);

            // Split heads: [b, s, d] -> [b, h, s, d/h]
            const splitHeads = (x) => x.reshape([batchSize, seqLen, this.numHeads, depth]).transpose([0, 2, 1, 3]);
            q = splitHeads(q);
            k = splitHeads(k);
            v = splitHeads(v);

            // Scaled Dot-Product Attention
            let logits = q.matMul(k, false, true).div(Math.sqrt(depth)); // [b, h, s, s]
            if (this.causal) {
                const mask = tf.linalg.bandPart(tf.ones([seqLen, seqLen]), -1, 0).sub(1).mul(1e9);
                const bmask = mask.expandDims(0).expandDims(0);
                logits = logits.add(bmask);
            }
            let weights = tf.softmax(logits, -1);
            let attn = weights.matMul(v); // [b, h, s, d/h]

            // Merge heads: [b, h, s, d/h] -> [b, s, d]
            attn = attn.transpose([0, 2, 1, 3]).reshape([batchSize, seqLen, this.embedDim]);

            // Output projection
            const attn2 = attn.reshape([-1, this.embedDim]);
            let out2 = attn2.matMul(this.oKernel.read());
            if (this.useBias) out2 = out2.add(this.oBias.read());
            const out = out2.reshape([batchSize, seqLen, this.embedDim]);
            return out;
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