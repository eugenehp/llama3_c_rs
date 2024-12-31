const BUFFER_TOKENS: i32 = 1;
const STATS: i32 = 1;
const LLAMA_VER: i32 = 2;
const ROPE_TF: f32 = 10000.0;
const BOS: i32 = 1;
const EOS: i32 = 2;
const GS: usize = 64; // Group size for quantization

static mut SYSTEM_TEMPLATE: [u8; 1024] = [0; 1024];
static mut USER_TEMPLATE: [u8; 1024] = [0; 1024];

struct Config {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
}

struct QuantizedTensor {
    q: Vec<i8>,
    s: Vec<f32>,
}

struct TransformerWeights {
    q_tokens: QuantizedTensor,
    token_embedding_table: Vec<f32>,
    rms_att_weight: Vec<f32>,
    rms_ffn_weight: Vec<f32>,
    wq: Vec<QuantizedTensor>,
    wk: Vec<QuantizedTensor>,
    wv: Vec<QuantizedTensor>,
    wo: Vec<QuantizedTensor>,
    w1: Vec<QuantizedTensor>,
    w2: Vec<QuantizedTensor>,
    w3: Vec<QuantizedTensor>,
    rms_final_weight: Vec<f32>,
    wcls: Option<QuantizedTensor>,
}

struct RunState {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    xq: QuantizedTensor,
    hq: QuantizedTensor,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

fn dequantize(qx: &QuantizedTensor, x: &mut [f32]) {
    for i in 0..x.len() {
        x[i] = qx.q[i] as f32 * qx.s[i / GS as usize];
    }
}

fn quantize(qx: &mut QuantizedTensor, x: &[f32]) {
    let num_groups = x.len() / GS as usize;
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

fn malloc_run_state(config: &Config) -> RunState {
    let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    RunState {
        x: vec![0.0; config.dim as usize],
        xb: vec![0.0; config.dim as usize],
        xb2: vec![0.0; config.dim as usize],
        hb: vec![0.0; config.hidden_dim as usize],
        hb2: vec![0.0; config.hidden_dim as usize],
        xq: QuantizedTensor {
            q: vec![0; config.dim as usize],
            s: vec![0.0; config.dim as usize],
        },
        hq: QuantizedTensor {
            q: vec![0; config.hidden_dim as usize],
            s: vec![0.0; config.hidden_dim as usize],
        },
        q: vec![0.0; config.dim as usize],
        k: vec![0.0; kv_dim as usize],
        v: vec![0.0; kv_dim as usize],
        att: vec![0.0; (config.n_heads * config.seq_len) as usize],
        logits: vec![0.0; config.vocab_size as usize],
        key_cache: vec![0.0; (config.n_layers * config.seq_len * kv_dim) as usize],
        value_cache: vec![0.0; (config.n_layers * config.seq_len * kv_dim) as usize],
    }
}

fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: usize) {
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

fn softmax(x: &mut [f32]) {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter().map(|&xi| (xi - max_val).exp()).sum();
    for xi in x.iter_mut() {
        *xi = (*xi - max_val).exp() / sum;
    }
}