### Load testing

Create a Locust load testing instance by using the CLI tool `load_test.py`

It *requires* locust to be installed, use `pip install locustio` (if dev-dependencies is not already installed).

#### Usage

- Start port-forwarding with *ambassador* on a cluster (with an existing gordo-project).

    `kubectl port-forward svc/ambassador -n ambassador 8888:80`

- Use the CLI tool `load_test.py`, specify the *project name* (required), *host* (optional) and *port* (optional).
**Note** the port needs to match the port which is port-forwarded to*. 
An example is which uses ambassador is: 
 
    `python load_test.py --project-name example-project --ambassador`


- Go to the Locust instance at: [http://127.0.0.1:8089](http://127.0.0.1:8089) and specify the number of users to simulate, as well as the the number of users
to *hatch* per second.
  

#### Advanced customization
Additional parameters can be specified in the script, such as *min-wait* and *max-wait*, as well as other Locust
parameters to prioritize certain tasks over others. Please see: [locust documentation](https://docs.locust.io/en/stable/)
for more information.
