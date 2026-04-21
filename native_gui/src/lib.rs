use core_network::P2pMessage;
use iced::{
    widget::{button, column, container, row, scrollable, text, text_input},
    Application, Command, Element, Length, Settings, Theme, subscription, Subscription
};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

pub fn run_gui(tx: mpsc::Sender<P2pMessage>, rx: Arc<Mutex<mpsc::Receiver<P2pMessage>>>) -> iced::Result {
    let mut settings = Settings::with_flags((tx, rx));
    settings.antialiasing = true;
    settings.window.size = iced::Size::new(1024.0, 768.0);
    SwarmNodeApp::run(settings)
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tab {
    Chat,
    Network,
    Settings,
}

struct SwarmNodeApp {
    current_tab: Tab,
    prompt: String,
    messages: Vec<String>,
    tx: mpsc::Sender<P2pMessage>,
    rx: Arc<Mutex<mpsc::Receiver<P2pMessage>>>,
    active_peers: usize,
}

#[derive(Debug, Clone)]
enum Message {
    TabSelected(Tab),
    PromptChanged(String),
    SubmitPrompt,
    NetworkEvent(P2pMessage),
}

impl Application for SwarmNodeApp {
    type Executor = iced::executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = (mpsc::Sender<P2pMessage>, Arc<Mutex<mpsc::Receiver<P2pMessage>>>);

    fn new(flags: Self::Flags) -> (Self, Command<Message>) {
        (
            Self {
                current_tab: Tab::Chat,
                prompt: String::new(),
                messages: vec!["System: SwarmNode Connected. Ready for P2P inference.".to_string()],
                tx: flags.0,
                rx: flags.1,
                active_peers: 0,
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("SwarmNode - Distributed AI Network")
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::TabSelected(tab) => {
                self.current_tab = tab;
                Command::none()
            }
            Message::PromptChanged(val) => {
                self.prompt = val;
                Command::none()
            }
            Message::SubmitPrompt => {
                if !self.prompt.is_empty() {
                    self.messages.push(format!("User: {}", self.prompt));
                    self.messages.push("Network: Routing inference to swarm...".to_string());
                    
                    let req = P2pMessage::InferenceRequest {
                        prompt: self.prompt.clone(),
                        max_tokens: 128,
                    };
                    self.prompt.clear();
                    
                    let tx = self.tx.clone();
                    Command::perform(async move {
                        let _ = tx.send(req).await;
                    }, |_| Message::PromptChanged("".to_string())) // Dummy mapping
                } else {
                    Command::none()
                }
            }
            Message::NetworkEvent(event) => {
                match event {
                    P2pMessage::InferenceResult { text } => {
                        self.messages.push(format!("Swarm: {}", text));
                    }
                    P2pMessage::PeerStatus { .. } => {
                        self.active_peers += 1;
                    }
                    _ => {}
                }
                Command::none()
            }
        }
    }

    fn subscription(&self) -> Subscription<Message> {
        struct NetworkReceiver;
        let rx = self.rx.clone();
        
        subscription::unfold(std::any::TypeId::of::<NetworkReceiver>(), rx, move |rx| async move {
            let msg = {
                let mut rx_lock = rx.lock().await;
                rx_lock.recv().await
            };
            
            if let Some(m) = msg {
                (Message::NetworkEvent(m), rx)
            } else {
                tokio::time::sleep(std::time::Duration::from_secs(86400)).await;
                (Message::NetworkEvent(P2pMessage::Ping), rx)
            }
        })
    }

    fn view(&self) -> Element<Message> {
        let nav_bar = row![
            button(text("Chat")).on_press(Message::TabSelected(Tab::Chat)).padding(10),
            button(text("Topology")).on_press(Message::TabSelected(Tab::Network)).padding(10),
            button(text("Settings")).on_press(Message::TabSelected(Tab::Settings)).padding(10),
        ].spacing(20).padding(20);

        let content: Element<Message> = match self.current_tab {
            Tab::Chat => {
                let mut chat_history = column![].spacing(15);
                for msg in &self.messages {
                    chat_history = chat_history.push(
                        container(text(msg).size(16))
                            .padding(15)
                            .style(iced::theme::Container::Box)
                    );
                }

                let chat_scroll = scrollable(chat_history).height(Length::Fill);

                let input_row = row![
                    text_input("Message the Swarm...", &self.prompt)
                        .on_input(Message::PromptChanged)
                        .on_submit(Message::SubmitPrompt)
                        .padding(15)
                        .size(16),
                    button(text("Send").size(16))
                        .on_press(Message::SubmitPrompt)
                        .padding(15)
                ].spacing(10);

                column![chat_scroll, input_row].spacing(20).into()
            },
            Tab::Network => {
                column![
                    text("Swarm Topology & Metrics").size(30),
                    text(format!("Active Peers Discovered: {}", self.active_peers)).size(20),
                    text("Local Transport: QUIC / TCP / mDNS Active").size(20),
                    text("Total Swarm VRAM: 24.0 GB (Estimated)").size(20),
                    text("Network Bandwidth: 14.2 MB/s").size(20),
                ].spacing(20).into()
            },
            Tab::Settings => {
                column![
                    text("Configuration").size(30),
                    text("Model: LLaMA-3-8B-Instruct.gguf (Local)").size(20),
                    button(text("Load Model (.gguf)")).padding(10),
                    text("Network Port: 4001 (Auto-NAT)").size(20),
                    text("Hardware Allocation: GPU (100%)").size(20),
                ].spacing(20).into()
            }
        };

        container(column![nav_bar, container(content).padding(20).width(Length::Fill).height(Length::Fill)])
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
    
    fn theme(&self) -> Theme {
        Theme::Dark
    }
}
