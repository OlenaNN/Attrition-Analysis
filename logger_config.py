import socket

from logging.config import dictConfig


def configure_logger(config):
    handlers = []

    if config.CONSOLE_LOGGING:
        handlers.append('console')

    log_config = {
        'version': 1,
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default'
            },
            'splunk': {
                'class': 'vfc.splunk.logging.SplunkHandler',
                'formatter': 'default',
                'source': config.APP_NAME,
                'sourcetype': config.SOURCE_TYPE,
                'hostname': socket.gethostname(),
            },
            'file': {
                'class': 'logging.FileHandler',
                'formatter': 'default',
                'filename': f'{config.APP_NAME}.log',
                'mode': 'w'  # clear the file on each launch
            }
        },
        'root': {
            'handlers': handlers,
        },
        'loggers': {
            'app': {
                'level': config.LOG_LEVEL,
            },
            'vfc': {
                'level': config.LOG_LEVEL,
            }
        }
    }
    dictConfig(log_config)
