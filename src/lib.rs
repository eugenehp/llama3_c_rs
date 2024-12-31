use rand::Rng;
use std::fs::File;
use std::io::Read;
use std::mem;
use std::os::unix::io::AsRawFd;
use std::slice; // For generating random values

const BUFFER_TOKENS: usize = 1;
const STATS: usize = 1;
const LLAMA_VER: usize = 2;
#[cfg(feature = "llama2")]
const ROPE_TF: f32 = 10_000.0;
#[cfg(feature = "llama3")]
const ROPE_TF: f32 = 50_0000.0;
const BOS: usize = 1;
const EOS: usize = 2;
const GS: usize = 64; // Group size for quantization

static mut SYSTEM_TEMPLATE: [u8; 1024] = [0; 1024];
static mut USER_TEMPLATE: [u8; 1024] = [0; 1024];

#[derive(Debug, Clone)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
}

#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub q: Vec<i8>,
    pub s: Vec<f32>,
}

#[derive(Debug)]
pub struct TransformerWeights {
    pub q_tokens: QuantizedTensor,
    pub token_embedding_table: Vec<f32>,
    pub rms_att_weight: Vec<f32>,
    pub rms_ffn_weight: Vec<f32>,
    pub wq: Vec<QuantizedTensor>,
    pub wk: Vec<QuantizedTensor>,
    pub wv: Vec<QuantizedTensor>,
    pub wo: Vec<QuantizedTensor>,
    pub w1: Vec<QuantizedTensor>,
    pub w2: Vec<QuantizedTensor>,
    pub w3: Vec<QuantizedTensor>,
    pub rms_final_weight: Vec<f32>,
    pub wcls: QuantizedTensor,
}

impl TransformerWeights {
    // Function to initialize weights based on the config
    pub fn from_config(config: &Config) -> Self {
        let mut rng = rand::thread_rng();

        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let n_layers = config.n_layers;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let seq_len = config.seq_len;
        let vocab_size = config.vocab_size;

        // Initialize token embedding table with random values
        let token_embedding_table: Vec<f32> = (0..vocab_size * dim)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        // Initialize RMS weights with random values (small values)
        let rms_att_weight: Vec<f32> = (0..n_layers * dim)
            .map(|_| rng.gen_range(0.0..0.1))
            .collect();
        let rms_ffn_weight: Vec<f32> = (0..n_layers * dim)
            .map(|_| rng.gen_range(0.0..0.1))
            .collect();
        let rms_final_weight: Vec<f32> = (0..dim).map(|_| rng.gen_range(0.0..0.1)).collect();

        // Initialize q_tokens as QuantizedTensor (using random values for q and s)
        let q_tokens = QuantizedTensor {
            q: vec![0; vocab_size * dim], // Initialize quantized values to 0
            s: vec![1.0; vocab_size],     // Initialize scale factors to 1
        };

        // Initialize the weight matrices (wq, wk, wv, etc.) as QuantizedTensors
        let wq = (0..n_layers)
            .map(|_| QuantizedTensor {
                q: vec![0; dim * dim],
                s: vec![1.0; dim],
            })
            .collect();

        let wk = (0..n_layers)
            .map(|_| QuantizedTensor {
                q: vec![0; dim * (dim / n_kv_heads)], // Key size depends on kv_heads
                s: vec![1.0; dim],
            })
            .collect();

        let wv = (0..n_layers)
            .map(|_| QuantizedTensor {
                q: vec![0; dim * (dim / n_kv_heads)], // Value size depends on kv_heads
                s: vec![1.0; dim],
            })
            .collect();

        let wo = (0..n_layers)
            .map(|_| QuantizedTensor {
                q: vec![0; dim * dim],
                s: vec![1.0; dim],
            })
            .collect();

        let w1 = (0..n_layers)
            .map(|_| QuantizedTensor {
                q: vec![0; dim * hidden_dim],
                s: vec![1.0; dim],
            })
            .collect();

        let w2 = (0..n_layers)
            .map(|_| QuantizedTensor {
                q: vec![0; hidden_dim * dim],
                s: vec![1.0; hidden_dim],
            })
            .collect();

        let w3 = (0..n_layers)
            .map(|_| QuantizedTensor {
                q: vec![0; dim * hidden_dim],
                s: vec![1.0; dim],
            })
            .collect();

        // Initialize wcls as a QuantizedTensor (classifier weights)
        let wcls = QuantizedTensor {
            q: vec![0; dim * vocab_size],
            s: vec![1.0; dim],
        };

        // Return the initialized TransformerWeights
        TransformerWeights {
            q_tokens,
            token_embedding_table,
            rms_att_weight,
            rms_ffn_weight,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_final_weight,
            wcls,
        }
    }
}

#[derive(Debug)]
pub struct RunState {
    pub x: Vec<f32>,
    pub xb: Vec<f32>,
    pub xb2: Vec<f32>,
    pub hb: Vec<f32>,
    pub hb2: Vec<f32>,
    pub xq: QuantizedTensor,
    pub hq: QuantizedTensor,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub att: Vec<f32>,
    pub logits: Vec<f32>,
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
}

#[derive(Debug)]
pub struct Transformer {
    pub config: Config,
    pub weights: TransformerWeights,
    pub state: RunState,
}

pub fn dequantize(qx: &QuantizedTensor, x: &mut [f32]) {
    for i in 0..x.len() {
        x[i] = qx.q[i] as f32 * qx.s[i / GS];
    }
}

pub fn quantize(qx: &mut QuantizedTensor, x: &[f32]) {
    let num_groups = x.len() / GS;
    let q_max = 127.0;
    for group in 0..num_groups {
        let mut wmax = 0.0;
        for i in 0..GS {
            let val = x[group * GS + i].abs();
            if val > wmax {
                wmax = val;
            }
        }
        let scale = wmax / q_max;
        qx.s[group] = scale;
        for i in 0..GS {
            let quant_value = x[group * GS + i] / scale;
            let quantized = quant_value.round() as i8;
            qx.q[group * GS + i] = quantized;
        }
    }
}

pub fn malloc_run_state(config: &Config) -> RunState {
    let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    RunState {
        x: vec![0.0; config.dim],
        xb: vec![0.0; config.dim],
        xb2: vec![0.0; config.dim],
        hb: vec![0.0; config.hidden_dim],
        hb2: vec![0.0; config.hidden_dim],
        xq: QuantizedTensor {
            q: vec![0; config.dim],
            s: vec![0.0; config.dim],
        },
        hq: QuantizedTensor {
            q: vec![0; config.hidden_dim],
            s: vec![0.0; config.hidden_dim],
        },
        q: vec![0.0; config.dim],
        k: vec![0.0; kv_dim],
        v: vec![0.0; kv_dim],
        att: vec![0.0; config.n_heads * config.seq_len],
        logits: vec![0.0; config.vocab_size],
        key_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
        value_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
    }
}

pub fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: usize) {
    let mut ss = 0.0;
    for j in 0..size {
        ss += x[j] * x[j];
    }
    ss /= size as f32;
    ss += 1e-5;
    ss = 1.0 / ss.sqrt();
    for j in 0..size {
        o[j] = weight[j] * (ss * x[j]);
    }
}

pub fn softmax(x: &mut [f32]) {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter().map(|&xi| (xi - max_val).exp()).sum();
    for xi in x.iter_mut() {
        *xi = (*xi - max_val).exp() / sum;
    }
}

pub fn matmul(xout: &mut [f32], x: &QuantizedTensor, w: &QuantizedTensor, n: usize, d: usize) {
    // W (d,n) @ x (n,) -> xout (d,)
    // The most time-consuming part of the function

    // Iterate over the rows of the result matrix
    for i in 0..d {
        let mut val = 0.0f32;
        let mut ival = 0i32;
        let in_offset = i * n;

        // Process in groups of GS
        for j in (0..n).step_by(GS) {
            // Unroll the inner loop by a factor of 4 (assuming GS is 4)
            for k in 0..GS {
                if j + k + 3 < n {
                    // Accumulate values for each 4-tuple
                    ival += (x.q[j + k] as i32) * (w.q[in_offset + j + k] as i32);
                }
            }
            // Scale the result according to scaling factors in `w.s` and `x.s`
            val += (ival as f32) * w.s[in_offset + j] * x.s[j];
            ival = 0; // Reset the accumulator for the next block
        }

        // Store the result in the output array
        xout[i] = val;
    }
}

pub fn forward(transformer: &mut Transformer, token: usize, pos: usize) -> &Vec<f32> {
    // Convenience variables
    let p = &transformer.config;
    let w = &transformer.weights;
    let s = &mut transformer.state;
    let dim = p.dim;
    let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    let kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier for kv sharing in multiquery
    let head_size = dim / p.n_heads;

    // Copy the token embedding into x
    s.x.copy_from_slice(&w.token_embedding_table[token * dim..(token + 1) * dim]);

    // Forward all the layers
    for l in 0..p.n_layers {
        // Attention rmsnorm
        rmsnorm(
            &mut s.xb,
            &s.x,
            &w.rms_att_weight[l * dim..(l + 1) * dim],
            dim,
        );

        // QKV matmuls for this position
        quantize(&mut s.xq, &s.xb);
        matmul(&mut s.q, &s.xq, &w.wq[l], dim, dim);
        matmul(&mut s.k, &s.xq, &w.wk[l], dim, kv_dim);
        matmul(&mut s.v, &s.xq, &w.wv[l], dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for i in (0..dim).step_by(2) {
            let head_dim = i % head_size;
            let freq = 1.0 / (ROPE_TF.powf(head_dim as f32 / head_size as f32));
            let val = pos as f32 * freq;
            let fcr = val.cos();
            let fci = val.sin();
            let rotn = if i < kv_dim { 2 } else { 1 };

            for v in 0..rotn {
                let vec = if v == 0 { &mut s.q } else { &mut s.k };
                let v0 = vec[i];
                let v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // Save key and value at this time step (pos) to kv cache
        let loff = l * p.seq_len * kv_dim; // KV cache layer offset for convenience
        let key_cache_row = &mut s.key_cache[loff + pos * kv_dim..loff + (pos + 1) * kv_dim];
        let value_cache_row = &mut s.value_cache[loff + pos * kv_dim..loff + (pos + 1) * kv_dim];
        key_cache_row.copy_from_slice(&s.k);
        value_cache_row.copy_from_slice(&s.v);

        // Multihead attention. Iterate over all heads
        for h in 0..p.n_heads {
            // Get the query vector for this head
            let q = &mut s.q[h * head_size..(h + 1) * head_size];
            // Attention scores for this head
            let att = &mut s.att[h * p.seq_len..(h + 1) * p.seq_len];

            // Iterate over all timesteps, including the current one
            for t in 0..=pos {
                // Get the key vector for this head and at this timestep
                let k = &s.key_cache[loff + t * kv_dim + (h / kv_mul) * head_size
                    ..loff + (t + 1) * kv_dim + (h / kv_mul) * head_size];
                // Calculate the attention score as the dot product of q and k
                let score: f32 = q.iter().zip(k.iter()).map(|(qi, ki)| qi * ki).sum();
                let score = score / (head_size as f32).sqrt();
                // Save the score to the attention buffer
                att[t] = score;
            }

            // Softmax the scores to get attention weights
            softmax(&mut att[0..=pos]);

            // Weighted sum of the values, store back into xb
            let xb = &mut s.xb[h * head_size..(h + 1) * head_size];
            xb.fill(0.0);

            for t in 0..=pos {
                // Get the value vector for this head and at this timestep
                let v = &s.value_cache[loff + t * kv_dim + (h / kv_mul) * head_size
                    ..loff + (t + 1) * kv_dim + (h / kv_mul) * head_size];
                // Get the attention weight for this timestep
                let a = att[t];
                // Accumulate the weighted value into xb
                for (i, vi) in xb.iter_mut().enumerate() {
                    *vi += a * v[i];
                }
            }
        }

        // Final matmul to get the output of the attention
        quantize(&mut s.xq, &s.xb);
        matmul(&mut s.xb2, &s.xq, &w.wo[l], dim, dim);

        // Residual connection back into x
        for i in 0..dim {
            s.x[i] += s.xb2[i];
        }

        // FFN rmsnorm
        rmsnorm(
            &mut s.xb,
            &s.x,
            &w.rms_ffn_weight[l * dim..(l + 1) * dim],
            dim,
        );

        // FFN layers
        quantize(&mut s.xq, &s.xb);
        matmul(&mut s.hb, &s.xq, &w.w1[l], dim, p.hidden_dim);
        matmul(&mut s.hb2, &s.xq, &w.w3[l], dim, p.hidden_dim);

        // SwiGLU non-linearity
        for i in 0..p.hidden_dim {
            let mut val = s.hb[i];
            val *= 1.0 / (1.0 + (-val).exp()); // Silu activation
            val *= s.hb2[i];
            s.hb[i] = val;
        }

        // Final matmul to get the output of the FFN
        quantize(&mut s.hq, &s.hb);
        matmul(&mut s.xb, &s.hq, &w.w2[l], p.hidden_dim, dim);

        // Residual connection
        for i in 0..dim {
            s.x[i] += s.xb[i];
        }
    }

    let sx = s.x.clone();

    // Final rmsnorm
    rmsnorm(&mut s.x, &sx, &w.rms_final_weight, dim);

    // Classifier into logits
    quantize(&mut s.xq, &s.x);
    matmul(&mut s.logits, &s.xq, &w.wcls, dim, p.vocab_size);

    &s.logits
}

pub fn memory_map_weights(
    weights: &mut TransformerWeights,
    config: &Config,
    ptr: *const u8,
    shared_classifier: bool,
) {
    let mut offset = 0;

    unsafe {
        let fptr = ptr as *const f32;

        // Map RMSNorm weights
        weights.rms_att_weight = Vec::from_raw_parts(
            fptr.add(offset) as *mut f32,
            config.n_layers as usize * config.dim as usize,
            config.n_layers as usize * config.dim as usize,
        );
        offset += config.n_layers as usize * config.dim as usize;

        weights.rms_ffn_weight = Vec::from_raw_parts(
            fptr.add(offset) as *mut f32,
            config.n_layers as usize * config.dim as usize,
            config.n_layers as usize * config.dim as usize,
        );
        offset += config.n_layers as usize * config.dim as usize;

        weights.rms_final_weight = Vec::from_raw_parts(
            fptr.add(offset) as *mut f32,
            config.dim as usize,
            config.dim as usize,
        );
        offset += config.dim as usize;

        // Map quantized token embeddings
        weights.q_tokens.q = Vec::from_raw_parts(
            ptr.add(offset) as *mut i8,
            config.vocab_size as usize * config.dim as usize,
            config.vocab_size as usize * config.dim as usize,
        );
        offset += config.vocab_size as usize * config.dim as usize;

        weights.q_tokens.s = Vec::from_raw_parts(
            ptr.add(offset) as *mut f32,
            config.vocab_size as usize * config.dim as usize / GS,
            config.vocab_size as usize * config.dim as usize / GS,
        );
        offset += config.vocab_size as usize * config.dim as usize / GS;

        // Dequantize token embedding table
        weights.token_embedding_table = vec![0.0; config.vocab_size as usize * config.dim as usize];
        dequantize(&weights.q_tokens, &mut weights.token_embedding_table);

        // Map other quantized tensors
        fn map_quantized_tensor(
            ptr: *const u8,
            offset: &mut usize,
            count: usize,
            size: usize,
        ) -> QuantizedTensor {
            unsafe {
                let q =
                    Vec::from_raw_parts(ptr.add(*offset) as *mut i8, count * size, count * size);
                *offset += count * size;

                let s = Vec::from_raw_parts(
                    ptr.add(*offset) as *mut f32,
                    count * size / GS,
                    count * size / GS,
                );
                *offset += count * size / GS;

                QuantizedTensor { q, s }
            }
        }

        weights.wq = (0..config.n_layers)
            .map(|_| {
                map_quantized_tensor(
                    ptr,
                    &mut offset,
                    config.dim as usize,
                    config.n_heads as usize * config.dim as usize / config.n_heads as usize,
                )
            })
            .collect();
        weights.wk = (0..config.n_layers)
            .map(|_| {
                map_quantized_tensor(
                    ptr,
                    &mut offset,
                    config.dim as usize,
                    config.n_kv_heads as usize * config.dim as usize / config.n_heads as usize,
                )
            })
            .collect();
        weights.wv = (0..config.n_layers)
            .map(|_| {
                map_quantized_tensor(
                    ptr,
                    &mut offset,
                    config.dim as usize,
                    config.n_kv_heads as usize * config.dim as usize / config.n_heads as usize,
                )
            })
            .collect();
        weights.wo = (0..config.n_layers)
            .map(|_| {
                map_quantized_tensor(
                    ptr,
                    &mut offset,
                    config.n_heads as usize * config.dim as usize / config.n_heads as usize,
                    config.dim as usize,
                )
            })
            .collect();
        weights.w1 = (0..config.n_layers)
            .map(|_| {
                map_quantized_tensor(
                    ptr,
                    &mut offset,
                    config.dim as usize,
                    config.hidden_dim as usize,
                )
            })
            .collect();
        weights.w2 = (0..config.n_layers)
            .map(|_| {
                map_quantized_tensor(
                    ptr,
                    &mut offset,
                    config.hidden_dim as usize,
                    config.dim as usize,
                )
            })
            .collect();
        weights.w3 = (0..config.n_layers)
            .map(|_| {
                map_quantized_tensor(
                    ptr,
                    &mut offset,
                    config.dim as usize,
                    config.hidden_dim as usize,
                )
            })
            .collect();

        // Map classifier weights
        weights.wcls = if shared_classifier {
            weights.q_tokens.clone()
        } else {
            map_quantized_tensor(
                ptr,
                &mut offset,
                config.dim as usize,
                config.vocab_size as usize,
            )
        };
    }
}

pub fn read_checkpoint(
    checkpoint: &str,
    config: &mut Config,
    weights: &mut TransformerWeights,
    fd: &mut i32,
    data: &mut Vec<f32>,
    file_size: &mut usize,
) {
    let mut file = File::open(checkpoint).expect("Couldn't open checkpoint file");
    *file_size = file.metadata().expect("Couldn't read metadata").len() as usize;

    let mut buffer = vec![0u8; *file_size];
    file.read_exact(&mut buffer)
        .expect("Couldn't read checkpoint file");

    unsafe {
        let ptr = buffer.as_ptr();

        // Read magic number
        let magic_number = *(ptr as *const u32);
        if magic_number != 0x616b3432 {
            panic!("Bad magic number");
        }

        // Read version number
        let version = *(ptr.add(4) as *const i32);
        if version != 2 {
            panic!("Bad version {}, need version 2", version);
        }

        // Read config
        let config_size = mem::size_of::<Config>();
        let config_ptr = ptr.add(8) as *const Config;
        *config = (*config_ptr).clone();

        // Read flags
        let shared_classifier = *(ptr.add(8 + config_size) as *const u8) != 0;
        let group_size = *(ptr.add(8 + config_size + 1) as *const i32);

        // TODO: fix me
        // Set group size globally
        // unsafe {
        //     GS = group_size as usize;
        // }

        // Memory map the weights
        let weights_ptr = ptr.add(8 + config_size + 1 + 4);
        memory_map_weights(weights, config, weights_ptr, shared_classifier);
    }

    *fd = file.as_raw_fd();
    *data = unsafe {
        slice::from_raw_parts(
            buffer.as_ptr() as *const f32,
            *file_size / mem::size_of::<f32>(),
        )
    }
    .to_vec();
}
