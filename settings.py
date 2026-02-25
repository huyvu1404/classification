import os
from dotenv import load_dotenv
load_dotenv()

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

CMS_USER_NAME = os.getenv("CMS_USER_NAME", "")
CMS_PASSWORD = os.getenv("CMS_PASSWORD", "")
CMS_URL = os.getenv("CMS_URL", "")
CMS_REFERER = os.getenv("CMS_REFERER", "")
QC_API_ENDPOINT = os.getenv("QC_API_ENDPOINT", "")