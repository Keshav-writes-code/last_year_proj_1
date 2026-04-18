use iced::widget::{Column, button, column, container, row, scrollable, text, text_input};
use iced::{Application, Command, Element, Length, Settings, Theme, executor};
use tokio::sync::mpsc;

use core_network::{Command as NetCmd, SwarmMessage};
use inference_engine::DistributedModel;

pub fn run_gui(
    cmd_tx: mpsc::Sender<NetCmd>,
    event_rx: std::sync::mpsc::Receiver<SwarmMessage>,
) -> iced::Result {
    SwarmNodeApp::run(Settings {
        id: None,
        window: iced::window::Settings::default(),
        flags: Flags { cmd_tx, event_rx },
        fonts: Vec::new(),
        default_font: iced::Font::default(),
        default_text_size: iced::Pixels(16.0),
        antialiasing: false,
    })
}

struct Flags {
    cmd_tx: mpsc::Sender<NetCmd>,
    event_rx: std::sync::mpsc::Receiver<SwarmMessage>,
}

#[derive(Debug, Clone)]
pub enum Message {
    TabSelected(Tab),
    ChatInputChanged(String),
    ModelPathChanged(String),
    LoadModel,
    SendMessage,
    Tick,
    BackendEvent(SwarmMessage),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Chat,
    Swarm,
    Settings,
}

struct SwarmNodeApp {
    current_tab: Tab,
    chat_input: String,
    model_path_input: String,
    chat_history: Vec<String>,
    peers: Vec<String>,
    #[allow(dead_code)]
    cmd_tx: mpsc::Sender<NetCmd>,
    event_rx: std::sync::mpsc::Receiver<SwarmMessage>,
    engine: DistributedModel,
}

impl Application for SwarmNodeApp {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = Flags;

    fn new(flags: Self::Flags) -> (Self, Command<Self::Message>) {
        (
            Self {
                current_tab: Tab::Settings, // Start on settings to load model
                chat_input: String::new(),
                model_path_input: String::new(),
                chat_history: Vec::new(),
                peers: Vec::new(),
                cmd_tx: flags.cmd_tx,
                event_rx: flags.event_rx,
                engine: DistributedModel::new(),
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        "SwarmNode - Native AI Swarm".into()
    }

    fn theme(&self) -> Theme {
        Theme::Dark
    }

    fn update(&mut self, message: Self::Message) -> Command<Self::Message> {
        match message {
            Message::TabSelected(tab) => {
                self.current_tab = tab;
            }
            Message::ChatInputChanged(val) => {
                self.chat_input = val;
            }
            Message::ModelPathChanged(val) => {
                self.model_path_input = val;
            }
            Message::LoadModel => {
                if !self.model_path_input.is_empty() {
                    match self.engine.load_gguf(&self.model_path_input) {
                        Ok(_) => self
                            .chat_history
                            .push("System: Model loaded successfully.".into()),
                        Err(e) => self
                            .chat_history
                            .push(format!("System Error: Failed to load model: {}", e)),
                    }
                }
            }
            Message::SendMessage => {
                if !self.chat_input.is_empty() && self.engine.is_loaded() {
                    self.chat_history.push(format!("You: {}", self.chat_input));

                    // Get initial embeddings
                    if let Ok(mut state) = self.engine.initial_embedding(&self.chat_input) {
                        // Process chunk locally (e.g. 16 layers)
                        if let Ok(processed) = self.engine.forward_chunk(&state, 16) {
                            state = processed;
                            self.chat_history.push(format!(
                                "System: Local compute step done (layer {}).",
                                state.current_layer
                            ));

                            // If done, output text
                            if self.engine.is_finished(&state) {
                                if let Ok(out) = self.engine.decode_logits(&state) {
                                    self.chat_history.push(format!("Swarm (Local): {}", out));
                                }
                            } else {
                                // Route rest to a peer
                                if let Some(peer_str) = self.peers.first() {
                                    if let Ok(peer_id) = peer_str.parse() {
                                        self.chat_history.push(format!(
                                            "System: Routing layer {} to peer {}...",
                                            state.current_layer, peer_str
                                        ));

                                        // Attempt to send
                                        let _ = self.cmd_tx.try_send(NetCmd::RouteCompute {
                                            to: peer_id,
                                            state,
                                            layers: 16,
                                        });
                                    }
                                } else {
                                    self.chat_history.push("System Error: No peers to complete inference. Falling back to local...".into());
                                    // Fallback to local
                                    if let Ok(processed_final) =
                                        self.engine.forward_chunk(&state, 100)
                                        && let Ok(out) = self.engine.decode_logits(&processed_final)
                                        {
                                            self.chat_history
                                                .push(format!("Swarm (Local Fallback): {}", out));
                                        }
                                }
                            }
                        }
                    }

                    self.chat_input.clear();
                } else if !self.engine.is_loaded() {
                    self.chat_history
                        .push("System: Please load a model in Settings first.".into());
                    self.chat_input.clear();
                }
            }
            Message::Tick => {
                while let Ok(msg) = self.event_rx.try_recv() {
                    match msg {
                        SwarmMessage::PeerConnected(p) => {
                            self.peers.push(p.to_string());
                            self.chat_history
                                .push(format!("System: Peer {} connected.", p));
                        }
                        SwarmMessage::PeerDisconnected(p) => {
                            self.peers.retain(|x| x != &p.to_string());
                            self.chat_history
                                .push(format!("System: Peer {} disconnected.", p));
                        }
                        SwarmMessage::ComputeRequest {
                            from,
                            state,
                            layers,
                        } => {
                            if self.engine.is_loaded() {
                                self.chat_history.push(format!(
                                    "System: Computing {} layers for peer {}.",
                                    layers, from
                                ));
                                if let Ok(processed) = self.engine.forward_chunk(&state, layers) {
                                    // Normally we would send Result back, here we just simulate
                                    if self.engine.is_finished(&processed)
                                        && let Ok(out) = self.engine.decode_logits(&processed) {
                                            self.chat_history.push(format!(
                                                "System: Completed peer inference! Decoded: {}",
                                                out
                                            ));
                                        }
                                }
                            }
                        }
                        SwarmMessage::ComputeResult { from, state } => {
                            if self.engine.is_finished(&state) {
                                if let Ok(out) = self.engine.decode_logits(&state) {
                                    self.chat_history
                                        .push(format!("Swarm (from {}): {}", from, out));
                                }
                            } else {
                                self.chat_history.push(format!("System: Received intermediate state layer {} from {}. Continuing...", state.current_layer, from));
                                // Continue processing...
                            }
                        }
                    }
                }
            }
            Message::BackendEvent(_) => {}
        }
        Command::none()
    }

    fn view(&self) -> Element<'_, Self::Message> {
        let tabs = row![
            button("Chat").on_press(Message::TabSelected(Tab::Chat)),
            button("Swarm Topology").on_press(Message::TabSelected(Tab::Swarm)),
            button("Settings").on_press(Message::TabSelected(Tab::Settings)),
        ]
        .spacing(10);

        let content: Element<_> = match self.current_tab {
            Tab::Chat => {
                let mut history = Column::new().spacing(10);
                for msg in &self.chat_history {
                    history = history.push(text(msg));
                }

                let input = text_input("Type a message...", &self.chat_input)
                    .on_input(Message::ChatInputChanged)
                    .on_submit(Message::SendMessage);

                column![
                    scrollable(history).height(Length::Fill),
                    row![input, button("Send").on_press(Message::SendMessage)].spacing(10)
                ]
                .into()
            }
            Tab::Swarm => {
                let mut peer_list = Column::new().spacing(10);
                peer_list =
                    peer_list.push(text(format!("Connected Peers: {}", self.peers.len())).size(24));
                for p in &self.peers {
                    peer_list = peer_list.push(text(p));
                }
                peer_list.into()
            }
            Tab::Settings => {
                let path_input = text_input("/path/to/model.gguf", &self.model_path_input)
                    .on_input(Message::ModelPathChanged)
                    .on_submit(Message::LoadModel);

                column![
                    text("Configuration").size(24),
                    text("Local LLM Path (.gguf):"),
                    row![path_input, button("Load").on_press(Message::LoadModel)].spacing(10),
                    text(if self.engine.is_loaded() {
                        "Status: Model Loaded (Active)"
                    } else {
                        "Status: No Model Loaded"
                    }),
                    text("Port Configuration: Auto (mDNS)"),
                    text("Network Fallback: Enabled"),
                ]
                .spacing(10)
                .into()
            }
        };

        container(column![tabs, content].spacing(20))
            .padding(20)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    fn subscription(&self) -> iced::Subscription<Self::Message> {
        iced::time::every(std::time::Duration::from_millis(100)).map(|_| Message::Tick)
    }
}
