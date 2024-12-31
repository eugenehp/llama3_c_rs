use llama3_c::*;
#[cfg(feature = "openmp")]
use openmp_sys::{
    omp_get_num_procs, omp_get_supported_active_levels, omp_set_max_active_levels,
    omp_set_num_threads,
};

use std::env;
use std::process::exit;
use std::time::{SystemTime, UNIX_EPOCH};

fn read_prompt() -> String {
    use std::io::{self, Write};

    print!("\n{} ", std::env::consts::OS);
    io::stdout().flush().unwrap();

    let mut prompt = String::new();
    io::stdin().read_line(&mut prompt).unwrap();
    prompt.trim().to_string()
}

fn error_usage() -> ! {
    eprintln!("Usage: <program> <checkpoint_path> [-t temperature] [-p topp] [-s rng_seed] [-n steps] [-i prompt] [-z tokenizer_path] [-m mode] [-y system_prompt] [-b buffertokens] [-x stats] [-l llamaver]");
    exit(1);
}

fn main() {
    #[cfg(feature = "llama2")]
    let mut rope_tf: f32 = 10_000.0;
    #[cfg(feature = "llama3")]
    let mut rope_tf: f32 = 50_0000.0;
    // L2E Addition
    #[cfg(feature = "openmp")]
    {
        let num_threads = unsafe { omp_get_num_procs() }; // get the number of CPU cores
        unsafe { omp_set_num_threads(num_threads) }; // set the number of threads to use for parallel regions
        let num_levels = unsafe { omp_get_supported_active_levels() }; // get maximum number of nested parallel regions supported
        unsafe { omp_set_max_active_levels(num_levels) }; // set to maximum supported parallel regions
    }
    // END L2E Addition

    // Default parameters
    let mut checkpoint_path: Option<String> = None; // e.g. out/model.bin
    let mut tokenizer_path = "tokenizer.bin".to_string();
    let mut temperature = 1.0f32; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    let mut topp = 0.9f32; // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    let mut steps = 256; // number of steps to run for
    let mut prompt: Option<String> = None; // prompt string
    let mut rng_seed = 0u64; // seed rng with time by default
    let mut mode = "generate".to_string(); // generate|chat
    let mut system_prompt: Option<String> = None; // the (optional) system prompt to use in chat mode
    let mut buffertokens = 1;
    let mut stats = 1;

    // L2E Addition
    #[cfg(any(feature = "cosmo_zip", feature = "inc_bin", feature = "strlit"))]
    {
        // We read the embedded checkpoint from within the executable
        #[cfg(feature = "unik")]
        {
            println!("\n*** GURU UNMEDITATION :: BOOT > LLAMA HAS AWAKENED ***\n\n");
        }
        #[cfg(feature = "cosmo_zip")]
        {
            checkpoint_path = Some("/zip/out/model.bin".to_string());
            tokenizer_path = "/zip/tokenizer.bin".to_string();
        }
        #[cfg(any(feature = "inc_bin", feature = "strlit"))]
        {
            checkpoint_path = Some(emb_model_data.to_string());
            tokenizer_path = emb_tokenizer_data.to_string();
        }
        buffertokens = 8;
        #[cfg(feature = "lloop")]
        {
            stats = LOOPSTATUS;
            loop {
                // Start of loop
                prompt = Some(read_prompt());
                if prompt.as_deref() == Some("exit") {
                    exit(0);
                }
                // Process prompt...
            }
        }
    }
    // END L2E Addition

    // Poor man's Rust argparse to override the defaults above from the command line
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 {
        checkpoint_path = Some(args[1].clone());
    } else {
        error_usage();
    }

    let mut i = 2;
    while i < args.len() {
        if i + 1 >= args.len() || !args[i].starts_with('-') || args[i].len() != 2 {
            error_usage();
        }

        match args[i].chars().nth(1).unwrap() {
            't' => temperature = args[i + 1].parse().unwrap_or_else(|_| error_usage()),
            'p' => topp = args[i + 1].parse().unwrap_or_else(|_| error_usage()),
            's' => rng_seed = args[i + 1].parse().unwrap_or_else(|_| error_usage()),
            'n' => steps = args[i + 1].parse().unwrap_or_else(|_| error_usage()),
            'i' => prompt = Some(args[i + 1].clone()),
            'z' => tokenizer_path = args[i + 1].clone(),
            'm' => mode = args[i + 1].clone(),
            'y' => system_prompt = Some(args[i + 1].clone()),
            'b' => buffertokens = args[i + 1].parse().unwrap_or_else(|_| error_usage()),
            'x' => stats = args[i + 1].parse().unwrap_or_else(|_| error_usage()),
            'l' => {
                if args[i + 1].parse::<i32>().unwrap_or_else(|_| error_usage()) == 3 {
                    rope_tf = 500000.0;
                }
            }
            _ => error_usage(),
        }

        i += 2;
    }

    // Parameter validation/overrides
    if rng_seed == 0 {
        rng_seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
    if temperature < 0.0 {
        temperature = 0.0;
    }
    if topp < 0.0 || topp > 1.0 {
        topp = 0.9;
    }
    if steps < 0 {
        steps = 0;
    }

    // Build the Transformer via the model .bin file
    let mut transformer = Transformer::default();
    transformer.build_transformer(checkpoint_path.as_deref().unwrap());

    if steps == 0 || steps > transformer.config.seq_len {
        steps = transformer.config.seq_len;
    }

    // Build the Tokenizer via the tokenizer .bin file
    let mut tokenizer = Tokenizer::default();
    tokenizer.build_tokenizer(&tokenizer_path, transformer.config.vocab_size);

    // Build the Sampler
    let mut sampler = Sampler::default();
    sampler.build(transformer.config.vocab_size, temperature, topp, rng_seed);

    // Run!
    if mode == "generate" {
        transformer.generate(
            &tokenizer,
            &sampler,
            prompt.as_deref().unwrap_or_default(),
            steps,
        );
    } else if mode == "chat" {
        transformer.chat(
            &tokenizer,
            &sampler,
            prompt.as_deref().unwrap_or_default(),
            system_prompt.as_deref().unwrap_or_default(),
            steps,
        );
    } else {
        eprintln!("unknown mode: {}", mode);
        error_usage();
    }
}
