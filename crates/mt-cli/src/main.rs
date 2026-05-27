//! `movie-translator` binary entry point.
//!
//! Port of `movie_translator/main.py`: routes the first argument to a
//! subcommand — `extract` → extract, `iphone` → iphone, anything else → the
//! default translate command.
//!
//! Uses a multi-threaded tokio runtime: `run_all` overlaps file IO/CPU work
//! across worker threads while serialising GPU work on a single worker.

use clap::Parser;

use mt_cli::commands::{extract, iphone, translate};

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let argv: Vec<String> = std::env::args().collect();

    // Subcommand routing mirrors main.py: dispatch on the first positional.
    let code = match argv.get(1).map(String::as_str) {
        Some("extract") => {
            // clap parses argv[0] (prog) + the args after `extract`.
            let args = parse_or_exit::<extract::ExtractArgs>(&argv, 2, "extract");
            extract::run(args).await
        }
        Some("iphone") => {
            let args = parse_or_exit::<iphone::IphoneArgs>(&argv, 2, "iphone");
            iphone::run(args).await
        }
        _ => {
            // Default = translate. Parse the full argv (no subcommand to skip).
            let args = parse_or_exit::<translate::TranslateArgs>(&argv, 1, "movie-translator");
            translate::run(args).await
        }
    };

    std::process::exit(code);
}

/// Parse a subcommand's args from `argv` starting after `skip` tokens.
///
/// On parse error (including `--help`/`--version`) clap prints to the right
/// stream and we exit with its suggested code — matching argparse's behaviour.
fn parse_or_exit<T: Parser>(argv: &[String], skip: usize, prog: &str) -> T {
    // Rebuild an argv where index 0 is the program name clap expects.
    let mut rebuilt: Vec<String> = Vec::with_capacity(argv.len());
    rebuilt.push(prog.to_string());
    rebuilt.extend(argv.iter().skip(skip).cloned());

    match T::try_parse_from(&rebuilt) {
        Ok(args) => args,
        Err(e) => {
            // clap's Error::exit() prints help/errors and exits with the right code.
            e.exit();
        }
    }
}
