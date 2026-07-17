use {
    async_trait::async_trait,
    solana_tpu_client_next::{
        ConnectionWorkersSchedulerError, connection_workers_scheduler::WorkersBroadcaster,
        transaction_batch::TransactionBatch, workers_cache::WorkersCache,
    },
    std::net::SocketAddr,
};

/// [`BackpressuredBroadcaster`] sends transactions to all the workers, awaiting
/// free capacity on each worker's channel instead of dropping the batch when
/// the channel is full (as the default `NonblockingBroadcaster` does).
///
/// Note: leaders in the fanout are served sequentially, so a particularly slow
/// leader delays delivery to the remaining leaders in the same batch.
pub struct BackpressuredBroadcaster;

#[async_trait]
impl WorkersBroadcaster for BackpressuredBroadcaster {
    async fn send_to_workers(
        &self,
        workers: &mut WorkersCache,
        leaders: &[SocketAddr],
        transaction_batch: TransactionBatch,
    ) -> Result<(), ConnectionWorkersSchedulerError> {
        for leader in leaders {
            // Unlike `try_send_transactions_to_address`, this awaits until the
            // worker channel has room, so a full channel never surfaces as an
            // error and the batch is not dropped.
            let send_res = workers
                .send_transactions_to_address(leader, transaction_batch.clone())
                .await;
            if let Err(err) = send_res {
                log::debug!("Failed to send transactions to {leader:?}, worker send error: {err}.");
                // `send_transactions_to_address` already evicts the worker on
                // `ReceiverDropped`; the remaining errors (`WorkerNotFound`,
                // `ShutdownError`) are transient/expected and non-fatal here.
            }
        }
        Ok(())
    }
}
