use std::thread;
use tokio::sync::mpsc;

use core_network::{Command, NetworkTask, SwarmMessage};
use native_gui::run_gui;

fn main() -> iced::Result {
    // Setup channels for communication between the background P2P network and the UI
    let (cmd_tx, cmd_rx) = mpsc::channel::<Command>(32);
    let (event_tx, event_rx_async) = mpsc::channel::<SwarmMessage>(100);

    // We need a standard sync channel to poll from iced's subscription tick smoothly
    let (sync_event_tx, sync_event_rx) = std::sync::mpsc::channel::<SwarmMessage>();

    // Spawn a dedicated thread for the tokio async runtime
    thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async move {
            // Forward async events to the sync channel for the GUI
            tokio::spawn(async move {
                let mut rx = event_rx_async;
                while let Some(msg) = rx.recv().await {
                    if sync_event_tx.send(msg).is_err() {
                        break;
                    }
                }
            });

            match NetworkTask::new(cmd_rx, event_tx) {
                Ok(network_task) => {
                    network_task.run().await;
                }
                Err(e) => {
                    eprintln!("Failed to initialize P2P network: {}", e);
                }
            }
        });
    });

    // Run the native GUI on the main thread
    run_gui(cmd_tx, sync_event_rx)
}
