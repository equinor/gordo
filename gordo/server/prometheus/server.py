from prometheus_client import make_wsgi_app
from .metrics import create_registry

app = make_wsgi_app(create_registry())
