use kube::{
    api::v1Event,
    api::{Api, Informer, WatchEvent},
    client::APIClient,
    config,
};
use log::{info, warn};

fn main() -> Result<(), failure::Error> {
    std::env::set_var("RUST_LOG", "info,kube=trace");
    env_logger::init();
    let config = config::load_kube_config().expect("failed to load kubeconfig");
    let client = APIClient::new(config);

    let namespace = std::env::var("NAMESPACE").unwrap_or("kubeflow".into());

    let events = Api::v1Event(client).within(&namespace);
    let ei = Informer::new(events).init()?;

    loop {
        ei.poll()?;

        while let Some(event) = ei.pop() {
            handle_event(event)?;
        }
    }
}

fn handle_event(ev: WatchEvent<v1Event>) -> Result<(), failure::Error> {
    match ev {
        WatchEvent::Added(o) => {
            info!("New Event: {}, {}", o.type_, o.message);
        }
        WatchEvent::Modified(o) => {
            info!("Modified Event: {}", o.reason);
        }
        WatchEvent::Deleted(o) => {
            info!("Deleted Event: {}", o.message);
        }
        WatchEvent::Error(e) => {
            warn!("Error event: {:?}", e);
        }
    }
    Ok(())
}
