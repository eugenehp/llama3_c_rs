use rand::Rng;
use std::fs::File;
use std::io::{BufRead, BufReader};
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

#[derive(Debug, Clone)]
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
    pub fd: i32,
    pub data: Vec<f32>,
    pub file_size: usize,
    pub rope_tf: f32,
}

impl TransformerWeights {
    // Function to initialize weights based on the config
    pub fn from_config(config: &Config) -> Self {
        let mut rng = rand::thread_rng();

        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let n_layers = config.n_layers;
        // let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        // let seq_len = config.seq_len;
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

#[derive(Debug, Clone)]
pub struct TokenIndex {
    pub str: String, // The string representation of the token
    pub id: usize,   // The token's unique identifier or index in the vocabulary
}

#[derive(Debug, Clone)]
pub struct ProbIndex {
    pub prob: f32,    // The probability of the token
    pub index: usize, // The token's index in the vocabulary
}

pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub vocab_scores: Vec<f32>,
    pub sorted_vocab: Vec<TokenIndex>,
    pub vocab_size: usize,
    pub max_token_length: usize,
}

// Implementing the Drop trait for Tokenizer
impl Drop for Tokenizer {
    fn drop(&mut self) {
        // Clear the contents of the tokenizer fields
        self.vocab.clear();
        self.vocab_scores.clear();
        self.sorted_vocab.clear();
        self.vocab_size = 0;
        self.max_token_length = 0;
        println!("Tokenizer dropped and resources freed.");
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self {
            vocab: vec![],
            vocab_scores: vec![],
            sorted_vocab: vec![],
            vocab_size: 0,
            max_token_length: 0,
        }
    }
}

impl TokenIndex {
    pub fn new(str: String, id: usize) -> Self {
        TokenIndex { str, id }
    }
}

// Example function to build a sorted vocabulary
pub fn build_sorted_vocab(vocab: Vec<String>) -> Vec<TokenIndex> {
    let mut sorted_vocab: Vec<TokenIndex> = vocab
        .into_iter()
        .enumerate()
        .map(|(id, str)| TokenIndex::new(str, id))
        .collect();
    sorted_vocab.sort_by(|a, b| a.str.cmp(&b.str));
    sorted_vocab
}

impl ProbIndex {
    pub fn new(prob: f32, index: usize) -> Self {
        ProbIndex { prob, index }
    }
}

// Example function to perform top-p sampling
pub fn sample_topp(probabilities: &[f32], topp: f32) -> usize {
    let mut probindex: Vec<ProbIndex> = probabilities
        .iter()
        .enumerate()
        .map(|(index, &prob)| ProbIndex::new(prob, index))
        .collect();
    probindex.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap());

    let mut cumulative_prob = 0.0;
    for i in 0..probindex.len() {
        cumulative_prob += probindex[i].prob;
        if cumulative_prob > topp {
            return probindex[i].index;
        }
    }
    probindex.last().unwrap().index
}

pub fn build_tokenizer(tokenizer: &mut Tokenizer, tokenizer_path: &str, vocab_size: usize) {
    // Initialize the tokenizer from a file or embedded data

    // Open the tokenizer file
    let file = File::open(tokenizer_path).expect("Couldn't open tokenizer file");
    let mut reader = BufReader::new(file);

    // Read the maximum token length
    let mut max_token_length = String::new();
    reader
        .read_line(&mut max_token_length)
        .expect("Couldn't read max token length");
    tokenizer.max_token_length = max_token_length
        .trim()
        .parse()
        .expect("Invalid max token length");

    // Initialize the vocab and vocab_scores vectors
    tokenizer.vocab = Vec::with_capacity(vocab_size);
    tokenizer.vocab_scores = Vec::with_capacity(vocab_size);

    // Read the vocab tokens and their scores
    for _ in 0..vocab_size {
        // Read the score
        let mut score = String::new();
        reader
            .read_line(&mut score)
            .expect("Couldn't read vocab score");
        let score: f32 = score.trim().parse().expect("Invalid vocab score");

        // Read the token
        let mut token = String::new();
        reader
            .read_line(&mut token)
            .expect("Couldn't read vocab token");
        let token = token.trim().to_string();

        // Add the token and score to the tokenizer
        tokenizer.vocab.push(token);
        tokenizer.vocab_scores.push(score);
    }

    // Create the sorted_vocab for efficient lookup
    tokenizer.sorted_vocab = tokenizer
        .vocab
        .iter()
        .enumerate()
        .map(|(id, str)| TokenIndex {
            str: str.clone(),
            id,
        })
        .collect();

    // Sort the vocab by the string representation of the tokens
    tokenizer.sorted_vocab.sort_by(|a, b| a.str.cmp(&b.str));
}

pub fn encode(tokenizer: &Tokenizer, text: &str, bos: bool, eos: bool) -> Vec<usize> {
    let mut tokens = Vec::new();

    // Add BOS token if required
    if bos {
        tokens.push(1); // Assuming 1 is the BOS token ID
    }

    // Tokenize the text
    let mut buffer = String::new();
    for c in text.chars() {
        buffer.push(c);

        // Check if the buffer matches any token in the vocabulary
        if buffer.len() > tokenizer.max_token_length {
            buffer.clear();
            tokens.push(3); // Assuming 3 is the ID for unknown tokens
            continue;
        }

        if let Some(token_index) = tokenizer.sorted_vocab.iter().find(|&ti| ti.str == buffer) {
            tokens.push(token_index.id);
            buffer.clear();
        }
    }

    // Add remaining buffer as tokens
    if !buffer.is_empty() {
        for byte in buffer.bytes() {
            tokens.push((byte + 3) as usize); // Adjusting for custom byte token IDs
        }
    }

    // Add EOS token if required
    if eos {
        tokens.push(2); // Assuming 2 is the EOS token ID
    }

    tokens
}

pub fn decode(tokenizer: &Tokenizer, prev_token: usize, token: usize) -> String {
    let piece = &tokenizer.vocab[token];

    // Strip leading whitespace after BOS token if necessary
    if prev_token == 1 && piece.starts_with(' ') {
        return piece[1..].to_string();
    }

    if piece.starts_with("<0x") && piece.ends_with('>') && piece.len() == 6 {
        if let Ok(byte_val) = u8::from_str_radix(&piece[3..5], 16) {
            return (byte_val as char).to_string();
        }
    }

    piece.clone()
}
pub struct Sampler {
    pub probindex: Vec<ProbIndex>,
    pub temperature: f32,
    pub topp: f32,
    pub rng_state: u64,
}

impl Drop for Sampler {
    fn drop(&mut self) {
        // Clear the contents of the sampler fields
        self.probindex.clear();
        println!("Sampler dropped and resources freed.");
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            probindex: vec![],
            temperature: 1.0,
            topp: 0.9,
            rng_state: 0,
        }
    }
}

impl Sampler {
    /// Initialize the sampler with the required parameters
    pub fn build(&mut self, vocab_size: usize, temperature: f32, topp: f32, rng_seed: u64) {
        self.probindex = Vec::with_capacity(vocab_size);
        self.temperature = temperature;
        self.topp = topp;
        self.rng_state = rng_seed;
    }

    /// Sample the next token based on logits and sampling strategy
    pub fn sample(&mut self, logits: &[f32]) -> usize {
        // Apply the temperature to the logits
        let mut adjusted_logits: Vec<f32> = logits
            .iter()
            .map(|&logit| logit / self.temperature)
            .collect();

        // Apply softmax to the logits to get probabilities
        Self::softmax(&mut adjusted_logits);

        // Flip a (float) coin (this is our source of entropy for sampling)
        let coin = self.random_f32();

        // Sample from this distribution to get the next token
        if self.topp <= 0.0 || self.topp >= 1.0 {
            // Simply sample from the predicted probability distribution
            Self::sample_mult(&adjusted_logits, coin)
        } else {
            // Top-p (nucleus) sampling, clamping the least likely tokens to zero
            self.sample_topp(&adjusted_logits, coin)
        }
    }

    /// Apply softmax to logits to get probabilities
    fn softmax(logits: &mut [f32]) {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        for logit in logits.iter_mut() {
            *logit = (*logit - max_logit).exp() / sum;
        }
    }

    /// Sample index from probabilities (they must sum to 1!)
    fn sample_mult(probabilities: &[f32], coin: f32) -> usize {
        // Sample index from probabilities (they must sum to 1!)
        let mut cdf = 0.0;
        for (i, &prob) in probabilities.iter().enumerate() {
            cdf += prob;
            if coin < cdf {
                return i;
            }
        }
        probabilities.len() - 1 // In case of rounding errors
    }

    /// Top-p sampling
    fn sample_topp(&mut self, probabilities: &[f32], coin: f32) -> usize {
        // Top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp.

        self.probindex.clear();
        for (index, &prob) in probabilities.iter().enumerate() {
            self.probindex.push(ProbIndex { prob, index });
        }

        // Sort indices in descending order of probabilities
        self.probindex
            .sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap());

        // Truncate the list where cumulative probability exceeds topp
        let mut cumulative_prob = 0.0;
        let mut last_idx = self.probindex.len() - 1; // In case of rounding errors
        for (i, probindex) in self.probindex.iter().enumerate() {
            cumulative_prob += probindex.prob;
            if cumulative_prob > self.topp {
                last_idx = i;
                break;
            }
        }

        // Sample from the truncated list
        let r = coin * cumulative_prob;
        let mut cdf = 0.0;
        for i in 0..=last_idx {
            cdf += self.probindex[i].prob;
            if r < cdf {
                return self.probindex[i].index;
            }
        }
        self.probindex[last_idx].index // In case of rounding errors
    }

    /// Random float32 in [0,1)
    fn random_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state >> 12;
        self.rng_state ^= self.rng_state << 25;
        self.rng_state ^= self.rng_state >> 27;
        (self.rng_state as f32 / u64::MAX as f32).abs()
    }
}

impl Default for Transformer {
    fn default() -> Self {
        Self {
            config: Config {
                dim: 768,
                hidden_dim: 3072,
                n_layers: 12,
                n_heads: 12,
                n_kv_heads: 12,
                vocab_size: 50257,
                seq_len: 1024,
            },
            weights: TransformerWeights {
                q_tokens: QuantizedTensor {
                    q: vec![],
                    s: vec![],
                },
                token_embedding_table: vec![],
                rms_att_weight: vec![],
                rms_ffn_weight: vec![],
                wq: vec![],
                wk: vec![],
                wv: vec![],
                wo: vec![],
                w1: vec![],
                w2: vec![],
                w3: vec![],
                rms_final_weight: vec![],
                wcls: QuantizedTensor {
                    q: vec![],
                    s: vec![],
                },
            },
            state: RunState {
                x: vec![],
                xb: vec![],
                xb2: vec![],
                hb: vec![],
                hb2: vec![],
                xq: QuantizedTensor {
                    q: vec![],
                    s: vec![],
                },
                hq: QuantizedTensor {
                    q: vec![],
                    s: vec![],
                },
                q: vec![],
                k: vec![],
                v: vec![],
                att: vec![],
                logits: vec![],
                key_cache: vec![],
                value_cache: vec![],
            },
            fd: -1,
            data: vec![],
            file_size: 0,
            #[cfg(feature = "llama2")]
            rope_tf: 10_000.0,
            #[cfg(feature = "llama3")]
            rope_tf: 50_0000.0,
        }
    }
}

impl Transformer {
    pub fn build_transformer(&mut self, checkpoint_path: &str) {
        // Read in the Config and the Weights from the checkpoint
        self.read_checkpoint(checkpoint_path);

        // Allocate the RunState buffers
        self.malloc_run_state();
    }

    fn read_checkpoint(&mut self, checkpoint_path: &str) {
        use std::fs::File;
        use std::io::Read;
        use std::mem;

        let mut file = File::open(checkpoint_path).expect("Couldn't open checkpoint file");
        self.file_size = file.metadata().expect("Couldn't read metadata").len() as usize;

        let mut buffer = vec![0u8; self.file_size];
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
            self.config = (*config_ptr).clone();

            // Read flags
            let shared_classifier = *(ptr.add(8 + config_size) as *const u8) != 0;
            let group_size = *(ptr.add(8 + config_size + 1) as *const i32);

            // Memory map the weights
            let weights_ptr = ptr.add(8 + config_size + 1 + 4);
            self.memory_map_weights(weights_ptr, shared_classifier);
        }

        self.data = unsafe {
            std::slice::from_raw_parts(
                buffer.as_ptr() as *const f32,
                self.file_size / mem::size_of::<f32>(),
            )
        }
        .to_vec();
    }

    fn memory_map_weights(&mut self, ptr: *const u8, shared_classifier: bool) {
        let mut offset = 0;

        unsafe {
            let fptr = ptr as *const f32;

            // Map RMSNorm weights
            self.weights.rms_att_weight = Vec::from_raw_parts(
                fptr.add(offset) as *mut f32,
                self.config.n_layers as usize * self.config.dim as usize,
                self.config.n_layers as usize * self.config.dim as usize,
            );
            offset += self.config.n_layers as usize * self.config.dim as usize;

            self.weights.rms_ffn_weight = Vec::from_raw_parts(
                fptr.add(offset) as *mut f32,
                self.config.n_layers as usize * self.config.dim as usize,
                self.config.n_layers as usize * self.config.dim as usize,
            );
            offset += self.config.n_layers as usize * self.config.dim as usize;

            self.weights.rms_final_weight = Vec::from_raw_parts(
                fptr.add(offset) as *mut f32,
                self.config.dim as usize,
                self.config.dim as usize,
            );
            offset += self.config.dim as usize;

            // Map quantized token embeddings
            self.weights.q_tokens.q = Vec::from_raw_parts(
                ptr.add(offset) as *mut i8,
                self.config.vocab_size as usize * self.config.dim as usize,
                self.config.vocab_size as usize * self.config.dim as usize,
            );
            offset += self.config.vocab_size as usize * self.config.dim as usize;

            self.weights.q_tokens.s = Vec::from_raw_parts(
                ptr.add(offset) as *mut f32,
                self.config.vocab_size as usize * self.config.dim as usize / 64,
                self.config.vocab_size as usize * self.config.dim as usize / 64,
            );
            offset += self.config.vocab_size as usize * self.config.dim as usize / 64;

            // Dequantize token embedding table
            self.weights.token_embedding_table =
                vec![0.0; self.config.vocab_size as usize * self.config.dim as usize];

            let q_tokens = &self.weights.q_tokens;

            dequantize(&q_tokens, &mut self.weights.token_embedding_table);

            // Map other quantized tensors
            fn map_quantized_tensor(
                ptr: *const u8,
                offset: &mut usize,
                count: usize,
                size: usize,
            ) -> QuantizedTensor {
                unsafe {
                    let q = Vec::from_raw_parts(
                        ptr.add(*offset) as *mut i8,
                        count * size,
                        count * size,
                    );
                    *offset += count * size;

                    let s = Vec::from_raw_parts(
                        ptr.add(*offset) as *mut f32,
                        count * size / 64,
                        count * size / 64,
                    );
                    *offset += count * size / 64;

                    QuantizedTensor { q, s }
                }
            }

            self.weights.wq = (0..self.config.n_layers)
                .map(|_| {
                    map_quantized_tensor(
                        ptr,
                        &mut offset,
                        self.config.dim as usize,
                        self.config.n_heads as usize * self.config.dim as usize
                            / self.config.n_heads as usize,
                    )
                })
                .collect();
            self.weights.wk = (0..self.config.n_layers)
                .map(|_| {
                    map_quantized_tensor(
                        ptr,
                        &mut offset,
                        self.config.dim as usize,
                        self.config.n_kv_heads as usize * self.config.dim as usize
                            / self.config.n_heads as usize,
                    )
                })
                .collect();
            self.weights.wv = (0..self.config.n_layers)
                .map(|_| {
                    map_quantized_tensor(
                        ptr,
                        &mut offset,
                        self.config.dim as usize,
                        self.config.n_kv_heads as usize * self.config.dim as usize
                            / self.config.n_heads as usize,
                    )
                })
                .collect();
            self.weights.wo = (0..self.config.n_layers)
                .map(|_| {
                    map_quantized_tensor(
                        ptr,
                        &mut offset,
                        self.config.n_heads as usize * self.config.dim as usize
                            / self.config.n_heads as usize,
                        self.config.dim as usize,
                    )
                })
                .collect();
            self.weights.w1 = (0..self.config.n_layers)
                .map(|_| {
                    map_quantized_tensor(
                        ptr,
                        &mut offset,
                        self.config.dim as usize,
                        self.config.hidden_dim as usize,
                    )
                })
                .collect();
            self.weights.w2 = (0..self.config.n_layers)
                .map(|_| {
                    map_quantized_tensor(
                        ptr,
                        &mut offset,
                        self.config.hidden_dim as usize,
                        self.config.dim as usize,
                    )
                })
                .collect();
            self.weights.w3 = (0..self.config.n_layers)
                .map(|_| {
                    map_quantized_tensor(
                        ptr,
                        &mut offset,
                        self.config.dim as usize,
                        self.config.hidden_dim as usize,
                    )
                })
                .collect();

            // Map classifier weights
            self.weights.wcls = if shared_classifier {
                self.weights.q_tokens.clone()
            } else {
                map_quantized_tensor(
                    ptr,
                    &mut offset,
                    self.config.dim as usize,
                    self.config.vocab_size as usize,
                )
            };
        }
    }

    fn malloc_run_state(&mut self) {
        let kv_dim = (self.config.dim * self.config.n_kv_heads) / self.config.n_heads;
        self.state = RunState {
            x: vec![0.0; self.config.dim],
            xb: vec![0.0; self.config.dim],
            xb2: vec![0.0; self.config.dim],
            hb: vec![0.0; self.config.hidden_dim],
            hb2: vec![0.0; self.config.hidden_dim],
            xq: QuantizedTensor {
                q: vec![0; self.config.dim],
                s: vec![0.0; self.config.dim],
            },
            hq: QuantizedTensor {
                q: vec![0; self.config.hidden_dim],
                s: vec![0.0; self.config.hidden_dim],
            },
            q: vec![0.0; self.config.dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            att: vec![0.0; self.config.n_heads * self.config.seq_len],
            logits: vec![0.0; self.config.vocab_size],
            key_cache: vec![0.0; self.config.n_layers * self.config.seq_len * kv_dim],
            value_cache: vec![0.0; self.config.n_layers * self.config.seq_len * kv_dim],
        }
    }

    pub fn forward(&mut self, token: usize, pos: usize) -> &Vec<f32> {
        // Convenience variables
        let p = &self.config;
        let w = &self.weights;
        let dim = p.dim;
        let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        let kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier for kv sharing in multiquery
        let head_size = dim / p.n_heads;

        // Copy the token embedding into x
        self.state
            .x
            .copy_from_slice(&w.token_embedding_table[token * dim..(token + 1) * dim]);

        // Forward all the layers
        for l in 0..p.n_layers {
            // Attention rmsnorm
            rmsnorm(
                &mut self.state.xb,
                &self.state.x,
                &w.rms_att_weight[l * dim..(l + 1) * dim],
                dim,
            );

            // QKV matmuls for this position
            quantize(&mut self.state.xq, &self.state.xb);
            matmul(&mut self.state.q, &self.state.xq, &w.wq[l], dim, dim);
            matmul(&mut self.state.k, &self.state.xq, &w.wk[l], dim, kv_dim);
            matmul(&mut self.state.v, &self.state.xq, &w.wv[l], dim, kv_dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for i in (0..dim).step_by(2) {
                let head_dim = i % head_size;
                let freq = 1.0 / (self.rope_tf.powf(head_dim as f32 / head_size as f32));
                let val = pos as f32 * freq;
                let fcr = val.cos();
                let fci = val.sin();
                let rotn = if i < kv_dim { 2 } else { 1 };

                for v in 0..rotn {
                    let vec = if v == 0 {
                        &mut self.state.q
                    } else {
                        &mut self.state.k
                    };
                    let v0 = vec[i];
                    let v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // Save key and value at this time step (pos) to kv cache
            let loff = l * p.seq_len * kv_dim; // KV cache layer offset for convenience
            let key_cache_row =
                &mut self.state.key_cache[loff + pos * kv_dim..loff + (pos + 1) * kv_dim];
            let value_cache_row =
                &mut self.state.value_cache[loff + pos * kv_dim..loff + (pos + 1) * kv_dim];
            key_cache_row.copy_from_slice(&self.state.k);
            value_cache_row.copy_from_slice(&self.state.v);

            // Multihead attention. Iterate over all heads
            for h in 0..p.n_heads {
                // Get the query vector for this head
                let q = &mut self.state.q[h * head_size..(h + 1) * head_size];
                // Attention scores for this head
                let att = &mut self.state.att[h * p.seq_len..(h + 1) * p.seq_len];

                // Iterate over all timesteps, including the current one
                for t in 0..=pos {
                    // Get the key vector for this head and at this timestep
                    let k = &self.state.key_cache[loff + t * kv_dim + (h / kv_mul) * head_size
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
                let xb = &mut self.state.xb[h * head_size..(h + 1) * head_size];
                xb.fill(0.0);

                for t in 0..=pos {
                    // Get the value vector for this head and at this timestep
                    let v = &self.state.value_cache[loff + t * kv_dim + (h / kv_mul) * head_size
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
            quantize(&mut self.state.xq, &self.state.xb);
            matmul(&mut self.state.xb2, &self.state.xq, &w.wo[l], dim, dim);

            // Residual connection back into x
            for i in 0..dim {
                self.state.x[i] += self.state.xb2[i];
            }

            // FFN rmsnorm
            rmsnorm(
                &mut self.state.xb,
                &self.state.x,
                &w.rms_ffn_weight[l * dim..(l + 1) * dim],
                dim,
            );

            // FFN layers
            quantize(&mut self.state.xq, &self.state.xb);
            matmul(
                &mut self.state.hb,
                &self.state.xq,
                &w.w1[l],
                dim,
                p.hidden_dim,
            );
            matmul(
                &mut self.state.hb2,
                &self.state.xq,
                &w.w3[l],
                dim,
                p.hidden_dim,
            );

            // SwiGLU non-linearity
            for i in 0..p.hidden_dim {
                let mut val = self.state.hb[i];
                val *= 1.0 / (1.0 + (-val).exp()); // Silu activation
                val *= self.state.hb2[i];
                self.state.hb[i] = val;
            }

            // Final matmul to get the output of the FFN
            quantize(&mut self.state.hq, &self.state.hb);
            matmul(
                &mut self.state.xb,
                &self.state.hq,
                &w.w2[l],
                p.hidden_dim,
                dim,
            );

            // Residual connection
            for i in 0..dim {
                self.state.x[i] += self.state.xb[i];
            }
        }

        let sx = self.state.x.clone();

        // Final rmsnorm
        rmsnorm(&mut self.state.x, &sx, &w.rms_final_weight, dim);

        // Classifier into logits
        quantize(&mut self.state.xq, &self.state.x);
        matmul(
            &mut self.state.logits,
            &self.state.xq,
            &w.wcls,
            dim,
            p.vocab_size,
        );

        self.state.logits.as_ref()
    }

    /// Generates text based on the input prompt.
    pub fn generate(
        &mut self,
        tokenizer: &Tokenizer,
        sampler: &mut Sampler,
        prompt: &str,
        steps: usize,
    ) {
        // Encode the prompt into tokens
        let mut tokens = encode(tokenizer, prompt, true, false);

        // Iterate for the specified number of steps
        for step in 0..steps {
            let pos = tokens.len() - 1;
            let logits = self.forward(tokens[pos], pos);
            let next_token = sampler.sample(&logits);
            tokens.push(next_token);

            // Stop if we reach EOS token
            if next_token == 2 {
                break;
            }
        }

        // Decode the tokens back into text
        let mut decoded_text = String::new();
        let mut prev_token = 1; // Start with BOS token
        for &token in &tokens {
            decoded_text.push_str(&decode(tokenizer, prev_token, token));
            prev_token = token;
        }

        println!("Generated text: {}", decoded_text);
    }

    /// Handles interactive chat sessions.
    pub fn chat(
        &mut self,
        tokenizer: &Tokenizer,
        sampler: &mut Sampler,
        cli_user_prompt: &str,
        cli_system_prompt: &str,
        steps: usize,
    ) {
        let mut conversation = String::new();

        loop {
            println!("User: {}", cli_user_prompt);
            conversation.push_str(cli_user_prompt);
            conversation.push_str(" ");

            // Encode the conversation into tokens
            let mut tokens = encode(tokenizer, &conversation, true, false);

            // Iterate for the specified number of steps
            for step in 0..steps {
                let pos = tokens.len() - 1;
                let logits = self.forward(tokens[pos], pos);
                let next_token = sampler.sample(&logits);
                tokens.push(next_token);

                // Stop if we reach EOS token
                if next_token == 2 {
                    break;
                }
            }

            // Decode the tokens back into text
            let mut decoded_text = String::new();
            let mut prev_token = 1; // Start with BOS token
            for &token in &tokens {
                decoded_text.push_str(&decode(tokenizer, prev_token, token));
                prev_token = token;
            }

            println!("System: {}", decoded_text);
            conversation.push_str(&decoded_text);
            conversation.push_str(" ");

            // Read new user input
            let mut user_input = String::new();
            std::io::stdin().read_line(&mut user_input).unwrap();
            let user_input = user_input.trim();
            if user_input == "exit" {
                break;
            }
            // cli_user_prompt = user_input;
        }
    }
}
