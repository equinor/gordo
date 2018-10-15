# -*- coding: utf-8 -*-

'''
CLI interfaces
'''

import logging
import os
from ast import literal_eval

import click
from gordo_components.builder import build_model

logger = logging.getLogger(__name__)


@click.group('gordo-components')
def gordo():
    """
    The main entry point for the CLI interface
    """
    pass


@click.command()
@click.argument('output-dir',
                default='/data',
                envvar='OUTPUT_DIR')
@click.argument('model-config',
                envvar='MODEL_CONFIG',
                default='{"type": "keras"}',
                type=literal_eval)
@click.argument('data-config',
                envvar='DATA_CONFIG',
                default='{"type": "influx"}',
                type=literal_eval)
def build(output_dir, model_config, data_config):
    """
    Build a model and deposit it into 'output_dir' given the appropriate config
    settings.
    """

    # We have access to MACHINE_NAME & TAGS env vars
    # probably need to refactor this more generically after mvp. 
    data_config['tag_list'] = data_config.get(
        'tag_list',
        literal_eval(os.environ.get('TAGS', '[]'))
    )
    data_config['machine_name'] = data_config.get(
        'machine_name', 
        os.environ.get('MACHINE_NAME')
    )

    logger.info('Building, output will be at: {}'.format(output_dir))
    logger.info('Model config: {}'.format(model_config))
    logger.info('Data config: {}'.format(data_config))

    build_model(output_dir=output_dir,
                model_config=model_config,
                data_config=data_config)
    logger.info('Successfully built model, and deposited at {}'.format(output_dir))
    return 0


gordo.add_command(build)
