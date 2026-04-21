use core_network::{NetworkManager, P2pMessage};
use inference_engine::DistributedEngine;
use native_gui::run_gui;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::{mpsc, Mutex};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    tracing::info!("Booting SwarmNode Workspace (Product Build)...");

    // Channels for bidirectional IPC between UI thread and Network Daemon
    let (tx_out, rx_out) = mpsc::channel::<P2pMessage>(100); // GUI -> Network
    let (tx_in, rx_in) = mpsc::channel::<P2pMessage>(100);   // Network -> GUI
    
    let rx_in_shared = Arc::new(Mutex::new(rx_in));

    // Initialize the core backend in a separate thread to prevent blocking the UI
    std::thread::spawn(move || {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            // Validate inference engine can boot
            match DistributedEngine::new() {
                Ok(_engine) => tracing::info!("Inference engine initialized successfully."),
                Err(e) => tracing::error!("Failed to init inference engine: {}", e),
            }

            // Start actual libp2p swarm
            let network = NetworkManager::new(rx_out, tx_in);
            if let Err(e) = network.start().await {
                tracing::error!("Network daemon failed: {}", e);
            }
        });
    });

    // Run the native GUI on the main OS thread (Wayland/X11/Metal requirement)
    tracing::info!("Starting Native GUI Engine...");
    run_gui(tx_out, rx_in_shared).map_err(|e| anyhow::anyhow!("GUI Error: {}", e))?;

    Ok(())
}
