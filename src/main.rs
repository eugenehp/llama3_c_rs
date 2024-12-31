use llama3_c::*;

fn main() {
    // Example Config
    let config = Config {
        dim: 512,
        hidden_dim: 2048,
        n_layers: 12,
        n_heads: 8,
        n_kv_heads: 8,
        vocab_size: 50257,
        seq_len: 1024,
    };

    // Create the TransformerWeights from the Config
    let weights = TransformerWeights::from_config(&config);

    let mut transformer = Transformer {
        config: config.clone(),
        weights,
        state: malloc_run_state(&config),
        // fd: -1,
        // data: vec![],
        // file_size: 0,
    };

    // Load model weights from checkpoint
    let checkpoint_path = "path/to/model.bin";
    let mut fd = 0;
    let mut data = vec![];
    let mut file_size = 0;

    read_checkpoint(
        checkpoint_path,
        &mut transformer.config,
        &mut transformer.weights,
        &mut fd,
        &mut data,
        &mut file_size,
    );

    // Perform a forward pass with a sample token and position
    let token = 0;
    let pos = 0;
    let logits = forward(&mut transformer, token, pos);

    // Print the logits
    println!("{:?}", logits);
}
