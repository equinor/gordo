from diagrams import Diagram, Cluster
from diagrams.onprem.compute import Server
from diagrams.oci.compute import Container, Functions
from diagrams.k8s.controlplane import API
from diagrams.k8s.others import CRD
from diagrams.k8s.compute import Job, Pod
from diagrams.k8s.storage import PV
from diagrams.custom import Custom

with Diagram("Gordo flow", filename="_static/architecture_diagram", outformat="png", show=False) as diag:
    with Cluster("K8s"):
        gordo = CRD("Gordo")
        model = CRD("Model")
        api = API("")
        with Cluster("gordo-controller"):
            Server("API")
            controller = Functions("Controller")
        with Cluster("gordo-server"):
            server_api = Server("API")
        dpl_job = Job("dpl")
        workflow = Custom("Workflow", "./argo_logo.png")
        gordo_volume = PV("")
        model_builder1 = Pod("model_builder1")
        model_builder2 = Pod("model_builder2")
    gordo >> api
    model >> api
    api >> controller
    controller >> dpl_job
    dpl_job >> workflow
    workflow >> model_builder1
    workflow >> model_builder2
    model_builder1 >> gordo_volume
    model_builder2 >> gordo_volume
    workflow >> server_api
    gordo_volume >> server_api
