import logging

#from vfc.logging.logger_config import configure_logger

from config import Config
from model_adjuster import ModelAdjuster

logger = logging.getLogger('app.workflow')

if __name__ == '__main__':
    config = Config()
    #configure_logger(config)
    ModelAdjuster(config).run()
