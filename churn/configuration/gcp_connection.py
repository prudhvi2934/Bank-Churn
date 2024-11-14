import os
from churn.constants import GCP_CONNECTION_PATH
from google.cloud import storage


class GCPservice:

    storage_client = None

    def __init__(self):

        if GCPservice.storage_client == None:
            _connection_json = GCP_CONNECTION_PATH
            if _connection_json is None:
                raise Exception(
                    f"{GCP_CONNECTION_PATH} path is invalid or empty")

            GCPservice.storage_client = storage.Client.from_service_account_json(
                _connection_json)

        self.storage_client = GCPservice.storage_client
