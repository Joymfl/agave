use {
    std::{future::Future, sync::OnceLock},
    tokio::{
        io::{self, AsyncWriteExt},
        task::JoinHandle,
    },
    tokio_util::sync::CancellationToken,
};

static CANCEL_TOKEN: OnceLock<CancellationToken> = OnceLock::new();
static SIGNAL_HANDLER: OnceLock<JoinHandle<io::Result<()>>> = OnceLock::new();

/// Process-wide cancellation token, cancelled on the first Ctrl + C.
pub fn cancel_token() -> &'static CancellationToken {
    CANCEL_TOKEN.get_or_init(CancellationToken::new)
}

/// On interrupt the future is dropped at its current await point and an error is
/// returned, so the caller unwinds through its normal error path rather than
/// having the process killed from under it.
/// Transactions already in flight may
/// still land, so callers are responsible for reporting what might be half-done.
pub async fn run_until_interrupted<T>(
    fut: impl Future<Output = Result<T, Box<dyn std::error::Error>>>,
) -> Result<T, Box<dyn std::error::Error>> {
    SIGNAL_HANDLER.get_or_init(spawn_signal_handler);
    tokio::select! {
        biased;
        () = cancel_token().cancelled() => {
            Err("Interrupted. The operation may be partially completed on-chain".into())
        }
        result = fut => result,
    }
}

// Up to the token handler to handle graceful shutdown
// TODO: doesn't handle sigterm or other signals
fn spawn_signal_handler() -> JoinHandle<io::Result<()>> {
    tokio::spawn(async move {
        let mut stderr = io::stderr();

        tokio::signal::ctrl_c().await?;
        stderr
            .write_all(b"\nshutting down gracefully (Ctrl + C again to force)\n")
            .await?;
        stderr.flush().await?;

        cancel_token().cancel();

        tokio::signal::ctrl_c().await?;
        stderr.write_all(b"\nforcing exit\n").await?;
        stderr.flush().await?;

        std::process::exit(130); // 128 + SIGINT
    })
}
