use std::time::Duration;
use mdns_sd::{ServiceDaemon, ServiceEvent};
use tracing::{info, warn};

use aegis_core::pb::federated_learning_client::FederatedLearningClient;
use aegis_core::pb::{GetGlobalModelRequest, SubmitWeightsRequest};

/// Discover the aggregator server using mDNS.
/// Returns the URI of the first discovered service, or `None` if it times out.
async fn discover_server(timeout: Duration) -> Option<String> {
    let mdns = ServiceDaemon::new().ok()?;
    let service_type = "_aegis._tcp.local.";

    let receiver = mdns.browse(service_type).ok()?;
    info!("Browsing for mDNS service: {}", service_type);

    let start = tokio::time::Instant::now();

    while start.elapsed() < timeout {
        if let Ok(event) = tokio::time::timeout(Duration::from_millis(500), receiver.recv_async()).await {
            if let Ok(ServiceEvent::ServiceResolved(info)) = event {
                // Return the first resolved service.
                // Depending on the network, we might get multiple IPs. We pick the first IPv4.
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    info!("Starting Aegis Edge Node...");
    let client_id = "client-uuid-placeholder"; // In reality use Uuid::new_v4().to_string()

    // 1. Local Network Discovery
    let mut target_uri = "http://127.0.0.1:50051".to_string(); // Fallback address
    if let Some(uri) = discover_server(Duration::from_secs(3)).await {
        target_uri = uri;
    } else {
        warn!("mDNS discovery timed out. Falling back to hardcoded address: {}", target_uri);
    }

    // 2. Connect via gRPC
    info!("Connecting to aggregator at {}...", target_uri);
    let mut client = match FederatedLearningClient::connect(target_uri).await {
        Ok(c) => c,
        Err(e) => {
            warn!("Failed to connect to aggregator: {}", e);
            return Err(e.into());
        }
    };

    // 3. Pull the Global Model
    info!("Pulling the global model...");
    let get_req = tonic::Request::new(GetGlobalModelRequest {
        client_id: client_id.to_string(),
    });
    let get_resp = client.get_global_model(get_req).await?;
    let model_info = get_resp.into_inner();
    info!(
        "Received global model version: {}",
        model_info.model_version
    );

    // 4. Simulate Local Training
    info!("Simulating local training on private data...");
    tokio::time::sleep(Duration::from_secs(2)).await;
    let dummy_weights = b"locally_trained_dummy_weights".to_vec();

    // 5. Submit Weights
    info!("Submitting weight updates...");
    let submit_req = tonic::Request::new(SubmitWeightsRequest {
        client_id: client_id.to_string(),
        model_version: model_info.model_version,
        weights: dummy_weights,
        data_points_count: 100, // Dummy training data count
    });
    let submit_resp = client.submit_weights(submit_req).await?;
    let submit_info = submit_resp.into_inner();

    if submit_info.success {
        info!("Successfully submitted weights: {}", submit_info.message);
    } else {
        warn!("Failed to submit weights: {}", submit_info.message);
    }

    info!("Edge Node execution completed.");
    Ok(())
}
