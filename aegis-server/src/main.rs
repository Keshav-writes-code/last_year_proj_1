use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, error};

use aegis_core::pb::federated_learning_server::{FederatedLearning, FederatedLearningServer};
use aegis_core::pb::{GetGlobalModelRequest, GetGlobalModelResponse, SubmitWeightsRequest, SubmitWeightsResponse};
use aegis_core::model::LanguageModel;
use mdns_sd::{ServiceDaemon, ServiceInfo};

/// The Secure Aggregator state.
/// Holds the current version and global weights.
#[derive(Debug, Default)]
struct AggregatorState {
    global_model_version: u64,
    global_model: LanguageModel,
    update_count: u64,
}

#[derive(Debug)]
pub struct AegisAggregator {
    state: Arc<Mutex<AggregatorState>>,
}

impl AegisAggregator {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(AggregatorState {
                global_model_version: 1,
                global_model: LanguageModel::new(),
                update_count: 0,
            })),
        }
    }
}

#[tonic::async_trait]
impl FederatedLearning for AegisAggregator {
    async fn submit_weights(
        &self,
        request: Request<SubmitWeightsRequest>,
    ) -> Result<Response<SubmitWeightsResponse>, Status> {
        let req = request.into_inner();
        info!(
            "Received weights from client: {}, model_version: {}, data_points: {}",
            req.client_id, req.model_version, req.data_points_count
        );

        // Deserialize incoming language model
        let local_model: LanguageModel = match bincode::deserialize(&req.weights) {
            Ok(m) => m,
            Err(e) => {
                error!("Failed to deserialize weights from {}: {}", req.client_id, e);
                return Ok(Response::new(SubmitWeightsResponse {
                    success: false,
                    message: "Failed to deserialize payload.".to_string(),
                }));
            }
        };

        let mut state = self.state.lock().unwrap();
        // Secure aggregation: merge the incoming frequencies into the global model
        state.global_model.merge(&local_model);

        state.update_count += 1;
        info!("Total updates received: {}. Global data points: {}", state.update_count, state.global_model.data_points);

        // Bump the global model version for every update to propagate changes immediately
        state.global_model_version += 1;
        info!("Global model updated to version: {}", state.global_model_version);

        let reply = SubmitWeightsResponse {
            success: true,
            message: "Weights aggregated successfully.".to_string(),
        };

        Ok(Response::new(reply))
    }

    async fn get_global_model(
        &self,
        request: Request<GetGlobalModelRequest>,
    ) -> Result<Response<GetGlobalModelResponse>, Status> {
        let req = request.into_inner();
        info!("Client {} requested the global model.", req.client_id);

        let state = self.state.lock().unwrap();

        let serialized_model = match bincode::serialize(&state.global_model) {
            Ok(bytes) => bytes,
            Err(e) => {
                error!("Failed to serialize global model: {}", e);
                return Err(Status::internal("Failed to serialize global model."));
            }
        };

        let reply = GetGlobalModelResponse {
            model_version: state.global_model_version,
            global_weights: serialized_model,
        };

        Ok(Response::new(reply))
    }
}

/// Helper function to register the mDNS service.
fn register_mdns(port: u16) -> anyhow::Result<ServiceDaemon> {
    let mdns = ServiceDaemon::new()?;
    let service_type = "_aegis._tcp.local.";
    let instance_name = "aegis_aggregator";
    // We bind to an unspecified IP to announce on all interfaces, but mDNS needs an IP.
    // For simplicity, we just announce a placeholder IP if needed, or rely on mDNS-SD finding the local IP.
    let host_name = "aegis-server.local.";
    let host_ipv4 = "127.0.0.1";
    let properties = [("version", "0.1.0")];

    let service_info = ServiceInfo::new(
        service_type,
        instance_name,
        host_name,
        host_ipv4,
        port,
        &properties[..],
    )?;

    mdns.register(service_info)?;
    info!("Registered mDNS service: {}", service_type);

    Ok(mdns)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize structured logging
    tracing_subscriber::fmt::init();

    info!("Starting Aegis Secure Aggregator...");

    let port: u16 = 50051;
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), port);
    let aggregator = AegisAggregator::new();

    // Spawn mDNS announcement in a blocking way or just keep the daemon alive.
    let mdns = register_mdns(port)?;

    info!("Listening for gRPC traffic on {}", addr);

    // Setup graceful shutdown
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();

    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("failed to listen for event");
        info!("Received Ctrl-C, shutting down...");
        let _ = tx.send(());
    });

    Server::builder()
        .add_service(FederatedLearningServer::new(aggregator))
        .serve_with_shutdown(addr, async {
            rx.await.ok();
        })
        .await?;

    info!("Server stopped gracefully.");

    // Unregister mDNS
    let service_type = "_aegis._tcp.local.";
    let instance_name = "aegis_aggregator";
    let fullname = format!("{}.{}", instance_name, service_type);
    let _ = mdns.unregister(&fullname);

    Ok(())
}
