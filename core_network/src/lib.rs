use libp2p::{
    core::upgrade::Version,
    futures::StreamExt,
    mdns, noise, swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux, PeerId, Swarm, SwarmBuilder, Transport,
};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{error, info};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TensorChunk {
    pub layer_id: u32,
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum P2pMessage {
    InferenceRequest { prompt: String, max_tokens: u32 },
    TensorState(TensorChunk),
    InferenceResult { text: String },
    PeerStatus { vram_gb: f32, compute_power: u32 },
    Ping,
}

#[derive(NetworkBehaviour)]
pub struct SwarmNodeBehavior {
    pub mdns: mdns::tokio::Behaviour,
    // For a fully production app, Kademlia and RequestResponse would be added here.
    // We use mDNS for local discovery as the baseline L1 layer.
}

pub struct NetworkManager {
    pub local_peer_id: PeerId,
    pub rx: mpsc::Receiver<P2pMessage>,
    pub tx: mpsc::Sender<P2pMessage>,
}

impl NetworkManager {
    pub fn new(rx: mpsc::Receiver<P2pMessage>, tx: mpsc::Sender<P2pMessage>) -> Self {
        Self {
            local_peer_id: PeerId::random(),
            rx,
            tx,
        }
    }

    pub async fn start(mut self) -> anyhow::Result<()> {
        info!("Starting core P2P network with PeerID: {}", self.local_peer_id);

        let mut swarm = libp2p::SwarmBuilder::with_new_identity()
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_quic()
            .with_behaviour(|key| {
                let mdns = mdns::tokio::Behaviour::new(
                    mdns::Config::default(),
                    key.public().to_peer_id(),
                )?;
                Ok(SwarmNodeBehavior { mdns })
            })?
            .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
            .build();

        swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;
        swarm.listen_on("/ip4/0.0.0.0/udp/0/quic-v1".parse()?)?;

        loop {
            tokio::select! {
                event = swarm.select_next_some() => match event {
                    SwarmEvent::Behaviour(SwarmNodeBehaviorEvent::Mdns(mdns::Event::Discovered(list))) => {
                        for (peer_id, multiaddr) in list {
                            info!("mDNS discovered a new peer: {peer_id} @ {multiaddr}");
                            // In production, we'd add to Kademlia routing table here
                        }
                    },
                    SwarmEvent::Behaviour(SwarmNodeBehaviorEvent::Mdns(mdns::Event::Expired(list))) => {
                        for (peer_id, _multiaddr) in list {
                            info!("mDNS discover peer has expired: {peer_id}");
                        }
                    },
                    SwarmEvent::NewListenAddr { address, .. } => {
                        info!("Local node is listening on {address}");
                    }
                    _ => {}
                },
                Some(msg) = self.rx.recv() => {
                    info!("Network sending message out to swarm: {:?}", msg);
                    match msg {
                        P2pMessage::InferenceRequest { prompt, .. } => {
                            // Production integration: distribute via Gossipsub or RequestResponse.
                            // Here we simulate the network loopback for UI reactivity.
                            let reply = P2pMessage::InferenceResult {
                                text: format!("Distributed Inference Result for: {}", prompt)
                            };
                            let _ = self.tx.send(reply).await;
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}
