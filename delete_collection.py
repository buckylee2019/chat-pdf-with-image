from pymilvus import utility
from pymilvus import connections
from dotenv import load_dotenv
import os
load_dotenv()

connections.connect(

  host = os.environ.get("MILVUS_HOST"),
  port = os.environ.get("MILVUS_PORT")
)
utility.drop_collection("Superthunder")
