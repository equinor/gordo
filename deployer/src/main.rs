use kube::{
    client::APIClient,
    config, api::{Api, Object, PostParams, RawApi, Reflector, Void},
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
    let reflector: Reflector<Gordo> = Reflector::raw(client.clone(), resource).init()?;

    loop {
        // Update state changes
        reflector.poll()?;

        // Read updates
        reflector.read()?
            .into_iter()
            .for_each(|crd: Gordo| {

                let jobs = Api::v1Job(client.clone())
                    .within(&namespace);

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
                                "image": &format!("gordo-infrastructure/gordo-deploy:{}", crd.spec.version),
                                "env": [
                                    {"name": "GORDO_CONFIG", "value": &crd.spec.info}
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
                jobs.create(&postparams, gdj_serialized).expect("Failed to launch job.");

            });
    }
}
