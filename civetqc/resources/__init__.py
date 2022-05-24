import json
from pkg_resources import resource_filename

with open(resource_filename(__name__, "config.json")) as file:
    config = json.load(file)