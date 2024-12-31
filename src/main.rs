use llama3_c::*;

fn main() {
    let mut transformer = Transformer {
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
    };

    transformer.build_transformer("checkpoint_path.bin");

    println!("{:?}", transformer);
}
