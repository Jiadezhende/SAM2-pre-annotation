import os

# Windows: disable torch.compile (requires Triton, which is not available on Windows)
if os.name == 'nt':
    os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import argparse
import json
import logging
import logging.config

from dotenv import load_dotenv
load_dotenv()

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'stream': 'ext://sys.stdout',
            'formatter': 'standard',
        }
    },
    'root': {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'handlers': ['console'],
        'propagate': True,
    }
})

from label_studio_ml.api import init_app
from model import SAM2VideoModel


_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')


def get_kwargs_from_config(config_path=_DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        config = json.load(f)
    assert isinstance(config, dict)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAM2 Video Label Studio ML Backend')
    parser.add_argument('-p', '--port', dest='port', type=int, default=int(os.getenv('PORT', 9090)))
    parser.add_argument('--host', dest='host', type=str, default='0.0.0.0')
    parser.add_argument('--kwargs', '--with', dest='kwargs', metavar='KEY=VAL', nargs='+',
                        type=lambda kv: kv.split('='),
                        help='Additional LabelStudioMLBase model initialization kwargs')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true')
    parser.add_argument('--log-level', dest='log_level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=None)
    parser.add_argument('--model-dir', dest='model_dir', default=os.path.dirname(__file__))
    parser.add_argument('--check', dest='check', action='store_true',
                        help='Validate model instance before launching server')
    parser.add_argument('--basic-auth-user', default=os.environ.get('ML_SERVER_BASIC_AUTH_USER'))
    parser.add_argument('--basic-auth-pass', default=os.environ.get('ML_SERVER_BASIC_AUTH_PASS'))

    args = parser.parse_args()

    if args.log_level:
        logging.root.setLevel(args.log_level)

    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def parse_kwargs():
        param = {}
        for k, v in (args.kwargs or []):
            if v.isdigit():
                param[k] = int(v)
            elif v in ('True', 'true'):
                param[k] = True
            elif v in ('False', 'false'):
                param[k] = False
            elif isfloat(v):
                param[k] = float(v)
            else:
                param[k] = v
        return param

    kwargs = get_kwargs_from_config()
    if args.kwargs:
        kwargs.update(parse_kwargs())

    if args.check:
        print(f'Checking "{SAM2VideoModel.__name__}" instance creation...')
        SAM2VideoModel(**kwargs)

    app = init_app(
        model_class=SAM2VideoModel,
        basic_auth_user=args.basic_auth_user,
        basic_auth_pass=args.basic_auth_pass,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)

else:
    # For gunicorn / uWSGI use
    app = init_app(model_class=SAM2VideoModel)
