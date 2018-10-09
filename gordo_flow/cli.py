# -*- coding: utf-8 -*-

'''
CLI interfaces
'''

import logging
import click
from ast import literal_eval
from gordo_flow.builder import build_model

logger = logging.getLogger(__name__)


@click.group('gordo-flow')
def gordo():
    pass


@click.command()
@click.argument('output-dir', 
                envvar='OUTPUT_DIR')
@click.argument('model-config', 
                envvar='MODEL_CONFIG', 
                default='{}',
                type=lambda v: isinstance(literal_eval(v), dict))
@click.argument('data-config',
                envvar='DATA_CONFIG',
                default='{}',
                type=lambda v: isinstance(literal_eval(v), dict))
def build(output_dir, model_config, data_config):
    """
    Build a model and deposit it into 'output_dir' given the appropriate config
    settings. 
    """
    logger.info('Building, output will be at: {}'.format(output_dir))
    logger.info('Model config: {}'.format(model_config))
    logger.info('Data config: {}'.format(data_config))

    build_model(output_dir=output_dir,
                model_config=model_config,
                data_config=data_config)
    return 0

gordo.add_command(build)
