use kube::api::Informer;
use kube::{
    api::{Api, Object, PostParams, RawApi, Void, WatchEvent},
    client::APIClient,
    config,
};
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize, Clone)]
pub struct GordoSpec {
    version: String,
    info: String,
}

type Gordo = Object<GordoSpec, Void>;

fn main() -> Result<(), failure::Error> {
    std::env::set_var("RUST_LOG", "info,kube=trace");
    env_logger::init();

    let config = config::load_kube_config().expect("failed to load kubeconfig");
    let client = APIClient::new(config);

    let namespace = std::env::var("NAMESPACE").unwrap_or("kubeflow".into());

    let resource = RawApi::customResource("gordo").within(&namespace);
    let informer: Informer<Gordo> = Informer::raw(client.clone(), resource).init()?;

    loop {
        // Update state changes
        informer.poll()?;

        while let Some(event) = informer.pop() {
            match event {
                WatchEvent::Added(gordo) => {
                    // Create the job.
                    let spec = json!({
                        "apiVersion": "batch/v1",
                        "kind": "Job",
                        "metadata": {
                            "name": "gordo-deploy-job"
                        },
                        "template": {
                            "metadata": {
                                "name": "gordo-deploy"
                            },
                            "spec": {
                                "containers": [{
                                    "name": "gordo-deploy",
                                    "image": &format!("gordo-infrastructure/gordo-deploy:{}", gordo.spec.version),
                                    "env": [
                                        {"name": "GORDO_CONFIG", "value": &gordo.spec.info}
                                    ]
                                }],
                                "restartPolicy": "Never"
                            }
                        }
                    });

                    let gdj_serialized = serde_json::to_vec(&spec).unwrap();
                    let postparams = PostParams::default();

                    // Send off job, later we can add support to watching the job if needed via `jobs.watch(..)`
                    info!("Launching job!");
                    let jobs = Api::v1Job(client.clone()).within(&namespace);
                    jobs.create(&postparams, gdj_serialized)
                        .expect("Failed to launch job.");
                }
                WatchEvent::Modified(gordo) => {
                    info!("Gordo resource modified: {:?}", gordo.metadata.name)
                }
                WatchEvent::Deleted(gordo) => {
                    info!("Gordo resource deleted: {:?}", gordo.metadata.name)
                }
                WatchEvent::Error(e) => info!("Gordo resource error: {:?}", e),
            }
        }
    }
}
