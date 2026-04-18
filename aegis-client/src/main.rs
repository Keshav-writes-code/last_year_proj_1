use std::time::Duration;
use std::fs;
use mdns_sd::{ServiceDaemon, ServiceEvent};
use tracing::{info, warn, error};
use eframe::egui;
use tokio::sync::mpsc;
use uuid::Uuid;
use serde::{Serialize, Deserialize};

use aegis_core::pb::federated_learning_client::FederatedLearningClient;
use aegis_core::pb::{GetGlobalModelRequest, SubmitWeightsRequest};
use aegis_core::model::ModelWeights;
use candle_core::{Device, Tensor, DType, Module};
use candle_nn::{linear, VarMap, VarBuilder, Optimizer, SGD};
use std::collections::HashMap;

const NOTES_FILE: &str = "aegis_notes.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Note {
    id: String,
    title: String,
    content: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct AppData {
    notes: Vec<Note>,
}

impl AppData {
    fn load() -> Self {
        if let Ok(data) = fs::read_to_string(NOTES_FILE) {
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    fn save(&self) {
        if let Ok(data) = serde_json::to_string_pretty(self) {
            let _ = fs::write(NOTES_FILE, data);
        }
    }
}

/// Messages sent from the background worker to the UI.
#[derive(Debug)]
enum WorkerMessage {
    StatusUpdate(String),
    TrainingProgress(u32, f32), // (epoch, loss)
    GlobalModelReceived(u64, ModelWeights),
    SyncComplete,
}

/// Messages sent from the UI to the background worker.
#[derive(Debug)]
enum UiMessage {
    SyncWeights(Vec<u8>, u64), // Serialized local weights, data points count
    TrainAndSync(Vec<String>), // Pass the local notes for the worker to train on
}

struct AegisApp {
    app_data: AppData,
    selected_note_idx: Option<usize>,
    status_text: String,
    global_model_version: u64,
    global_model: ModelWeights,
    is_syncing: bool,
    training_epoch: Option<u32>,
    training_loss: Option<f32>,
    client_id: String,
    ui_tx: mpsc::Sender<UiMessage>,
    worker_rx: mpsc::Receiver<WorkerMessage>,
}

impl AegisApp {
    fn new(_cc: &eframe::CreationContext<'_>, ui_tx: mpsc::Sender<UiMessage>, worker_rx: mpsc::Receiver<WorkerMessage>) -> Self {
        Self {
            app_data: AppData::load(),
            selected_note_idx: None,
            status_text: "Initializing...".to_string(),
            global_model_version: 0,
            global_model: ModelWeights::new(),
            is_syncing: false,
            training_epoch: None,
            training_loss: None,
            client_id: Default::default(),
            ui_tx,
            worker_rx,
        }
    }
}

async fn discover_server(mdns: &ServiceDaemon, timeout: Duration) -> Option<String> {
    let service_type = "_aegis._tcp.local.";

    let receiver = mdns.browse(service_type).ok()?;
    info!("Browsing for mDNS service: {}", service_type);

    let start = tokio::time::Instant::now();

    while start.elapsed() < timeout {
        if let Ok(event) = tokio::time::timeout(Duration::from_millis(500), receiver.recv_async()).await {
            if let Ok(ServiceEvent::ServiceResolved(info)) = event {
                for ip in info.get_addresses() {
                    let uri = format!("http://{}:{}", ip, info.get_port());
                    info!("Discovered Aegis aggregator at {}", uri);
                    return Some(uri);
                }
            }
        }
    }
    None
}

async fn run_background_worker(
    client_id: String,
    mut ui_rx: mpsc::Receiver<UiMessage>,
    worker_tx: mpsc::Sender<WorkerMessage>,
) {
    // Keep the daemon alive for the lifetime of the worker thread
    // to prevent "sending on a closed channel" errors from background threads.
    let mdns_daemon = ServiceDaemon::new().unwrap();

    let _ = worker_tx.send(WorkerMessage::StatusUpdate("Discovering Aggregator...".to_string())).await;

    let mut target_uri = "http://127.0.0.1:50051".to_string();
    if let Some(uri) = discover_server(&mdns_daemon, Duration::from_secs(3)).await {
        target_uri = uri;
    } else {
        warn!("mDNS discovery timed out. Falling back to hardcoded address: {}", target_uri);
    }

    let _ = worker_tx.send(WorkerMessage::StatusUpdate(format!("Connecting to {}...", target_uri))).await;

    let mut client = match FederatedLearningClient::connect(target_uri).await {
        Ok(c) => c,
        Err(e) => {
            let _ = worker_tx.send(WorkerMessage::StatusUpdate(format!("Connection Failed: {}", e))).await;
            return;
        }
    };

    let _ = worker_tx.send(WorkerMessage::StatusUpdate("Connected. Pulling Global Model...".to_string())).await;

    let get_req = tonic::Request::new(GetGlobalModelRequest {
        client_id: client_id.clone(),
    });

    let mut current_model_version = 0;
    match client.get_global_model(get_req).await {
        Ok(resp) => {
            let info = resp.into_inner();
            current_model_version = info.model_version;

            match bincode::deserialize::<ModelWeights>(&info.global_weights) {
                Ok(model) => {
                    let _ = worker_tx.send(WorkerMessage::GlobalModelReceived(info.model_version, model)).await;
                }
                Err(e) => {
                    let _ = worker_tx.send(WorkerMessage::StatusUpdate(format!("Failed to deserialize model: {}", e))).await;
                }
            }
        }
        Err(e) => {
            let _ = worker_tx.send(WorkerMessage::StatusUpdate(format!("Failed to pull model: {}", e))).await;
            return;
        }
    }

    // Main loop for handling sync requests
    while let Some(msg) = ui_rx.recv().await {
        match msg {
            UiMessage::SyncWeights(weights, data_points) => {
                let submit_req = tonic::Request::new(SubmitWeightsRequest {
                    client_id: client_id.clone(),
                    model_version: current_model_version,
                    weights,
                    data_points_count: data_points,
                });

                match client.submit_weights(submit_req).await {
                    Ok(resp) => {
                        let info = resp.into_inner();
                        if info.success {
                            info!("Successfully submitted weights.");
                            let _ = worker_tx.send(WorkerMessage::StatusUpdate("Sync Complete.".to_string())).await;
                            let _ = worker_tx.send(WorkerMessage::SyncComplete).await;
                        } else {
                            let _ = worker_tx.send(WorkerMessage::StatusUpdate(format!("Sync Failed: {}", info.message))).await;
                            let _ = worker_tx.send(WorkerMessage::SyncComplete).await;
                        }
                    }
                    Err(e) => {
                        let _ = worker_tx.send(WorkerMessage::StatusUpdate(format!("Submit Error: {}", e))).await;
                        let _ = worker_tx.send(WorkerMessage::SyncComplete).await;
                    }
                }
            }
            UiMessage::TrainAndSync(texts) => {
                let _ = worker_tx.send(WorkerMessage::StatusUpdate("Training local candle tensor model...".to_string())).await;

                let data_points = texts.iter().map(|t| t.len() as u64).sum::<u64>();

                let device = Device::Cpu;

                // Initialize model using VarMap
                let varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

                // Dummy objective: predicting input length normalized from character count
                // A real model would tokenize, embed, and predict
                let linear_model = match linear(1, 1, vb.pp("linear")) {
                    Ok(l) => l,
                    Err(_) => {
                        let _ = worker_tx.send(WorkerMessage::StatusUpdate("Failed to init candle model".to_string())).await;
                        continue;
                    }
                };

                let mut sgd = match SGD::new(varmap.all_vars(), 0.01) {
                    Ok(opt) => opt,
                    Err(_) => {
                        let _ = worker_tx.send(WorkerMessage::StatusUpdate("Failed to init SGD optimizer".to_string())).await;
                        continue;
                    }
                };

                let mut current_loss = 0.0;
                let epochs = 50;

                for epoch in 1..=epochs {
                    let mut epoch_loss = 0.0;
                    for text in &texts {
                        let feature = text.len() as f32 / 100.0; // Normalize
                        let target = feature * 2.0 + 1.0; // Dummy true function

                        let xs = Tensor::new(&[[feature]], &device).unwrap();
                        let ys = Tensor::new(&[[target]], &device).unwrap();

                        if let Ok(predictions) = linear_model.forward(&xs) {
                            if let Ok(loss) = candle_nn::loss::mse(&predictions, &ys) {
                                let _ = sgd.backward_step(&loss);
                                epoch_loss += loss.to_vec0::<f32>().unwrap_or(0.0);
                            }
                        }
                    }

                    current_loss = epoch_loss / texts.len().max(1) as f32;
                    let _ = worker_tx.send(WorkerMessage::TrainingProgress(epoch, current_loss)).await;
                    tokio::time::sleep(Duration::from_millis(20)).await; // Add slight delay to make progress visible in UI
                }

                // Extract trained tensors back into a HashMap
                let mut tensor_map = HashMap::new();
                for (name, var) in varmap.data().lock().unwrap().iter() {
                    tensor_map.insert(name.clone(), var.as_tensor().clone());
                }

                match ModelWeights::from_tensors(&tensor_map, data_points) {
                    Ok(weights_struct) => {
                        match bincode::serialize(&weights_struct) {
                            Ok(weights_payload) => {
                                let _ = worker_tx.send(WorkerMessage::StatusUpdate("Pushing weights via gRPC...".to_string())).await;
                                let submit_req = tonic::Request::new(SubmitWeightsRequest {
                                    client_id: client_id.clone(),
                                    model_version: current_model_version,
                                    weights: weights_payload,
                                    data_points_count: data_points,
                                });

                                match client.submit_weights(submit_req).await {
                                    Ok(resp) => {
                                        if resp.into_inner().success {
                                            let _ = worker_tx.send(WorkerMessage::StatusUpdate(format!("Tensor sync complete. Final Loss: {:.4}", current_loss))).await;
                                        }
                                    }
                                    Err(e) => {
                                        let _ = worker_tx.send(WorkerMessage::StatusUpdate(format!("Tensor sync error: {}", e))).await;
                                    }
                                }
                            }
                            Err(_) => {
                                let _ = worker_tx.send(WorkerMessage::StatusUpdate("Serialization error".to_string())).await;
                            }
                        }
                    }
                    Err(_) => {
                        let _ = worker_tx.send(WorkerMessage::StatusUpdate("Tensor conversion error".to_string())).await;
                    }
                }
                let _ = worker_tx.send(WorkerMessage::SyncComplete).await;
            }
        }
    }
}

impl eframe::App for AegisApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll for messages from the background worker
        while let Ok(msg) = self.worker_rx.try_recv() {
            match msg {
                WorkerMessage::StatusUpdate(status) => {
                    self.status_text = status;
                }
                WorkerMessage::TrainingProgress(epoch, loss) => {
                    self.training_epoch = Some(epoch);
                    self.training_loss = Some(loss);
                }
                WorkerMessage::GlobalModelReceived(version, model) => {
                    self.global_model_version = version;
                    self.global_model = model;
                    self.status_text = format!("Global Model v{} loaded.", version);
                }
                WorkerMessage::SyncComplete => {
                    self.is_syncing = false;
                    self.training_epoch = None;
                    self.training_loss = None;
                    self.status_text = "Sync Complete.".to_string();
                }
            }
        }

        egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(format!("Status: {}", self.status_text));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.is_syncing {
                        if let (Some(epoch), Some(loss)) = (self.training_epoch, self.training_loss) {
                            ui.label(format!("Epoch {}/50 (Loss: {:.4})", epoch, loss));
                            ui.add(egui::ProgressBar::new(epoch as f32 / 50.0).desired_width(100.0));
                        } else {
                            ui.add(egui::Spinner::new());
                            ui.label("Syncing...");
                        }
                    } else {
                        if ui.button("Sync to Global Model").clicked() {
                            self.is_syncing = true;
                            self.status_text = "Training local candle model...".to_string();

                            let mut texts = vec![];
                            for note in &self.app_data.notes {
                                if !note.content.is_empty() {
                                    texts.push(note.content.clone());
                                }
                            }

                            if let Err(e) = self.ui_tx.try_send(UiMessage::TrainAndSync(texts)) {
                                error!("Failed to send train message to worker: {}", e);
                                self.is_syncing = false;
                                self.status_text = "Error sending train request.".to_string();
                            }
                        }
                    }
                    ui.label(format!("Global Model Version: {}", self.global_model_version));
                });
            });
            ui.add_space(4.0);
        });

        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("Aegis Notes");
            if ui.button("+ New Note").clicked() {
                self.app_data.notes.push(Note {
                    id: Uuid::new_v4().to_string(),
                    title: "New Note".to_string(),
                    content: "".to_string(),
                });
                self.selected_note_idx = Some(self.app_data.notes.len() - 1);
                self.app_data.save();
            }
            ui.separator();

            let mut note_to_delete = None;
            egui::ScrollArea::vertical().show(ui, |ui| {
                for (idx, note) in self.app_data.notes.iter().enumerate() {
                    ui.horizontal(|ui| {
                        if ui.selectable_label(Some(idx) == self.selected_note_idx, &note.title).clicked() {
                            self.selected_note_idx = Some(idx);
                        }
                        if ui.button("X").clicked() {
                            note_to_delete = Some(idx);
                        }
                    });
                }
            });

            if let Some(idx) = note_to_delete {
                self.app_data.notes.remove(idx);
                self.selected_note_idx = None;
                self.app_data.save();
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(idx) = self.selected_note_idx {
                let mut save_required = false;

                if let Some(note) = self.app_data.notes.get_mut(idx) {
                    ui.horizontal(|ui| {
                        ui.label("Title:");
                        if ui.text_edit_singleline(&mut note.title).changed() {
                            save_required = true;
                        }
                    });
                    ui.separator();

                    // Extract the last word of the content for prediction
                    // Removed simple autocomplete to focus on the ML Tensor Federated structure
                    ui.label("Candle ML Tensor mode enabled. Training will extract local gradients.");

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        let text_edit = egui::TextEdit::multiline(&mut note.content)
                            .desired_width(f32::INFINITY)
                            .desired_rows(20);

                        let response = ui.add(text_edit);
                        if response.changed() {
                            save_required = true;
                        }
                    });
                }

                if save_required {
                    self.app_data.save();
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Select or create a note.");
                });
            }
        });

        // Request a repaint to ensure we keep polling the channel if needed
        ctx.request_repaint_after(Duration::from_millis(100));
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    info!("Starting Aegis Edge Node with GUI...");

    let (ui_tx, ui_rx) = mpsc::channel(32);
    let (worker_tx, worker_rx) = mpsc::channel(32);

    let client_id = Uuid::new_v4().to_string();

    // Spawn the tokio runtime in a background thread
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let worker_client_id = client_id.clone();
    std::thread::spawn(move || {
        rt.block_on(async {
            run_background_worker(worker_client_id, ui_rx, worker_tx).await;
        });
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([600.0, 400.0])
            .with_title("Aegis Client"),
        ..Default::default()
    };

    eframe::run_native(
        "Aegis Client",
        options,
        Box::new(|cc| {
            let mut app = AegisApp::new(cc, ui_tx, worker_rx);
            app.client_id = client_id;
            Ok(Box::new(app))
        }),
    ).map_err(|e| anyhow::anyhow!("eframe error: {}", e))
}
