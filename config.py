import os

import dotenv

#from version import current_version

dotenv.load_dotenv()


class Config:
   # PROJECT = os.path.basename(os.path.dirname(__file__))
    APP_NAME = 'model_adjuster'
    SOURCE_TYPE = 'model_adjuster'
    S3_BUCKET = 'gl-ml-training-model-adjuster'
    LABELED_DATA_FOLDER = 'labeled_data_folder'

    CONSOLE_LOGGING = bool(int(os.environ.get('CONSOLE_LOGGING', '1')))
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

    # API_BASE_URL = os.environ.get('API_BASE_URL')
    # USER_EMAIL = os.environ.get('USER_EMAIL')
    # USER_PASSWORD = os.environ.get('USER_PASSWORD')
    #
    # EMAILS = [
    #     email.strip() for email in os.environ.get('EMAILS', '').split(',')
    # ]
    #
    # REDIS_HOST = os.environ.get('REDIS_HOST')
    # REDIS_PORT = os.environ.get('REDIS_PORT')
    # REDIS_DATABASE = os.environ.get('REDIS_DATABASE')
    #
    # MAX_SUBSAMPLE_SIZE = int(os.environ.get('MAX_SUBSAMPLE_SIZE') or 100)
