use iced::widget::{Column, button, column, container, row, scrollable, text, text_input};
use iced::{Application, Command, Element, Length, Settings, Theme, executor};
use tokio::sync::mpsc;

use core_network::{Command as NetCmd, SwarmMessage};
use inference_engine::{InferenceEngine, TensorState};

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
    InputChanged(String),
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
    chat_history: Vec<String>,
    peers: Vec<String>,
    #[allow(dead_code)]
    cmd_tx: mpsc::Sender<NetCmd>,
    event_rx: std::sync::mpsc::Receiver<SwarmMessage>,
    engine: InferenceEngine,
}

impl Application for SwarmNodeApp {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = Flags;

    fn new(flags: Self::Flags) -> (Self, Command<Self::Message>) {
        (
            Self {
                current_tab: Tab::Chat,
                chat_input: String::new(),
                chat_history: Vec::new(),
                peers: Vec::new(),
                cmd_tx: flags.cmd_tx,
                event_rx: flags.event_rx,
                engine: InferenceEngine::new(),
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        "SwarmNode".into()
    }

    fn theme(&self) -> Theme {
        Theme::Dark
    }

    fn update(&mut self, message: Self::Message) -> Command<Self::Message> {
        match message {
            Message::TabSelected(tab) => {
                self.current_tab = tab;
            }
            Message::InputChanged(val) => {
                self.chat_input = val;
            }
            Message::SendMessage => {
                if !self.chat_input.is_empty() {
                    self.chat_history.push(format!("You: {}", self.chat_input));

                    // Simulate local processing before sending over network
                    let initial_state = TensorState::new_simulated(&self.chat_input);
                    if let Ok(processed) = self.engine.process_layer(&initial_state) {
                        self.chat_history.push(format!(
                            "Local compute step done (step {}).",
                            processed.step
                        ));

                        // Send to a peer if we have any
                        if let Some(peer) = self.peers.first() {
                            // In a real app we'd need to parse the peer ID, but this is a dummy structure
                            // For now we just log it
                            self.chat_history
                                .push(format!("Routing to peer {}...", peer));
                        }
                    }

                    self.chat_input.clear();
                }
            }
            Message::Tick => {
                // Poll backend events
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
                        SwarmMessage::TensorReceived { from, state } => {
                            // Process and finalize or send further
                            if let Ok(next_state) = self.engine.process_layer(&state) {
                                let output = self.engine.finalize(&next_state);
                                self.chat_history
                                    .push(format!("Swarm (from {}): {}", from, output));
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
                    .on_input(Message::InputChanged)
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
            Tab::Settings => column![
                text("Configuration").size(24),
                text("Port Configuration: Auto (mDNS)"),
                text("Local LLM Compute: Enabled"),
                text("Network Fallback: Enabled"),
            ]
            .spacing(10)
            .into(),
        };

        container(column![tabs, content].spacing(20))
            .padding(20)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    fn subscription(&self) -> iced::Subscription<Self::Message> {
        // A simple ticker to poll the crossbeam/mpsc channel
        iced::time::every(std::time::Duration::from_millis(100)).map(|_| Message::Tick)
    }
}
