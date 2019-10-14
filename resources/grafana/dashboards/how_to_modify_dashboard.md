To change the dashboard:

1. Export the changed dashboard as json from grafana.
2. Place the content of the exported json-file inside the "dashboard" key in machines.json.
3. Find the "id" key inside the value of "dashboard", and change it to null.
