import os

import app.version
import dotenv

#from version import current_version

dotenv.load_dotenv()


class Config:
    PROJECT = os.path.basename(os.path.dirname(__file__))
    VERSION = app.version.current_version
    APP_NAME = 'naas_midas_modeladjuster'
    SOURCE_TYPE = 'midas_modeladjuster'
    S3_BUCKET = os.environ.get('S3_BUCKET')
    DOMAIN_NAME = os.environ.get('DOMAIN_NAME')
    LABELED_DATA_FOLDER = os.environ.get('LABELED_DATA_FOLDER')

    GIT_COMMIT_HASH = os.environ.get("GIT_COMMIT")

    SPLUNK_HOST = os.environ.get('SPLUNK_HOST')
    SPLUNK_PORT = os.environ.get('SPLUNK_PORT')
    SPLUNK_TOKEN = os.environ.get('SPLUNK_TOKEN')
    SPLUNK_INDEX = os.environ.get('SPLUNK_INDEX')  # optional

    SPLUNK_LOGGING = bool(int(os.environ.get('SPLUNK_LOGGING', '1')))
    FILE_LOGGING = bool(int(os.environ.get('FILE_LOGGING', '1')))
    CONSOLE_LOGGING = bool(int(os.environ.get('CONSOLE_LOGGING', '1')))
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

    API_BASE_URL = os.environ.get('API_BASE_URL')
    USER_EMAIL = os.environ.get('USER_EMAIL')
    USER_PASSWORD = os.environ.get('USER_PASSWORD')

    EMAILS = [
        email.strip() for email in os.environ.get('EMAILS', '').split(',')
    ]

    REDIS_HOST = os.environ.get('REDIS_HOST')
    REDIS_PORT = os.environ.get('REDIS_PORT')
    REDIS_DATABASE = os.environ.get('REDIS_DATABASE')

    MAX_SUBSAMPLE_SIZE = int(os.environ.get('MAX_SUBSAMPLE_SIZE') or 100)
