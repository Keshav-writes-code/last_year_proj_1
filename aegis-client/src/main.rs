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
use aegis_core::model::LanguageModel;

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
    GlobalModelReceived(u64, LanguageModel),
    SyncComplete,
}

/// Messages sent from the UI to the background worker.
#[derive(Debug)]
enum UiMessage {
    SyncWeights(Vec<u8>, u64), // Serialized local weights, data points count
}

struct AegisApp {
    app_data: AppData,
    selected_note_idx: Option<usize>,
    status_text: String,
    global_model_version: u64,
    global_model: LanguageModel,
    is_syncing: bool,
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
            global_model: LanguageModel::new(),
            is_syncing: false,
            client_id: Default::default(),
            ui_tx,
            worker_rx,
        }
    }

    fn train_local_model(&self) -> LanguageModel {
        let mut local_model = LanguageModel::new();
        for note in &self.app_data.notes {
            local_model.train(&note.content);
        }
        local_model
    }
}

async fn discover_server(timeout: Duration) -> Option<String> {
    let mdns = ServiceDaemon::new().ok()?;
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
    let _ = worker_tx.send(WorkerMessage::StatusUpdate("Discovering Aggregator...".to_string())).await;

    let mut target_uri = "http://127.0.0.1:50051".to_string();
    if let Some(uri) = discover_server(Duration::from_secs(3)).await {
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

            match bincode::deserialize::<LanguageModel>(&info.global_weights) {
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
                WorkerMessage::GlobalModelReceived(version, model) => {
                    self.global_model_version = version;
                    self.global_model = model;
                    self.status_text = format!("Global Model v{} loaded.", version);
                }
                WorkerMessage::SyncComplete => {
                    self.is_syncing = false;
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
                        ui.add(egui::Spinner::new());
                        ui.label("Syncing...");
                    } else {
                        if ui.button("Sync to Global Model").clicked() {
                            self.is_syncing = true;
                            self.status_text = "Training local model...".to_string();

                            let local_model = self.train_local_model();
                            let data_points = local_model.data_points;

                            match bincode::serialize(&local_model) {
                                Ok(weights_payload) => {
                                    if let Err(e) = self.ui_tx.try_send(UiMessage::SyncWeights(weights_payload, data_points)) {
                                        error!("Failed to send sync message to worker: {}", e);
                                        self.is_syncing = false;
                                        self.status_text = "Error sending sync request.".to_string();
                                    } else {
                                        self.status_text = "Sending updates to Aggregator...".to_string();
                                    }
                                }
                                Err(e) => {
                                    error!("Failed to serialize features: {}", e);
                                    self.is_syncing = false;
                                    self.status_text = "Error serializing features.".to_string();
                                }
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
                    let last_word = note.content
                        .split_whitespace()
                        .last()
                        .unwrap_or("");

                    let prediction = if !last_word.is_empty() {
                        self.global_model.predict_next_word(last_word)
                    } else {
                        None
                    };

                    ui.horizontal(|ui| {
                        if let Some(pred) = &prediction {
                            ui.label(format!("Prediction: {}", pred));
                            if ui.button("Insert (Tab)").clicked() || ctx.input(|i| i.key_pressed(egui::Key::Tab)) {
                                if !note.content.ends_with(' ') && !note.content.is_empty() {
                                    note.content.push(' ');
                                }
                                note.content.push_str(pred);
                                note.content.push(' ');
                                save_required = true;
                            }
                        } else {
                            ui.label("Prediction: (Type to see suggestions)");
                        }
                    });

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
