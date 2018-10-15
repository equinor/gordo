# -*- coding: utf-8 -*-


class InfluxTagFetcher:
    """
    Fetch a tag from Azure datalake and put that data into a running
    Influx DB
    """
    def __init__(self, db_name='default'):
        self.db_name = db_name



import argparse
import json
import logging
import timeit

import multiprocessing
import os
import pandas as pd
from azure.datalake.store import lib
from azure.datalake.store.lib import DataLakeCredential
from influxdb import DataFrameClient

from ._influx.azure_statoil_walker import (
    read_df_from_azure, read_iroc_df_from_azure, walk_and_tag_azure, walk_azure
)
from ._influx.wind_json_walker import read_wind_json_as_df, walk_the_street

logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.WARN)


def write_to_influx(df, tags, host, port, user, password, db_name, batch_size=10000, time_precision='s'):
    """ 
    Writes a data-frame with tags to influx, with time_precision=s. 
    """
    print("Write DataFrame with Tags {}, with length: {}".format(tags, len(df)))
    client = DataFrameClient(host, port, user, password, db_name)
    if not client.write_points(df, db_name, tags, time_precision=time_precision, batch_size=batch_size):
        logger.error("Writing to influx failed for tags: {}".format(tags))


def write_to_influx_multiseries(df, global_tags, tag_columns, field_columns, host, port, user, password, db_name, batch_size=10000, time_precision='s'):
    """ 
    Writes a data-frame with tags to influx, with time_precision=s.
    This method deals with the situation where the dataframe contains data from many sensors,
    and the tags/ID info is in columns
    """
    print("Write MultiSeries DataFrame with length: {}".format(len(df)))
    client = DataFrameClient(host, port, user, password, db_name)
    if not client.write_points(df, db_name, global_tags, tag_columns=tag_columns, field_columns=field_columns,
                               time_precision=time_precision, batch_size=batch_size):
        logger.error("Writing to influx failed")


def azure_fetch_and_push_to_influx(token, azure_data_store, file_path, tag_map, influx_settings):
    """ 
    Reads the file in file_path from the azure datalake azure_data_store using access token token,
    attempts to read it as a csv into panda. Expects the csv to have 4 columns of types String, Number, Timestamp,
    and Number. Only the timestamp and number is used, in a timestamp -> number mapping. This is written to the
    influxdb provided in influx_settings. influx_settings is expected to be on the format
    (host, port, user, password, db_name, batch_size).
    """
    df = read_df_from_azure(azure_data_store, file_path, token)
    (host, port, user, password, db_name, batch_size) = influx_settings
    write_to_influx(df, tag_map, host, port, user, password, db_name, batch_size)


def azure_iroc_fetch_and_push_to_influx(token, azure_data_store, file_path, influx_settings):
    """ 
    Reads the file in file_path from the azure datalake azure_data_store using access token token,
    attempts to read it as a csv into panda. Expects the csv to have 4 columns of types String, Number, Timestamp,
    and Number. Only the timestamp and number is used, in a timestamp -> number mapping. This is written to the
    influxdb provided in influx_settings. influx_settings is expected to be on the format
    (host, port, user, password, db_name, batch_size).
    """
    df = read_iroc_df_from_azure(azure_data_store, file_path, token)
    (host, port, user, password, db_name, batch_size) = influx_settings

    tag_columns = None # All except field_columns will be regarded as tags.
    field_columns = ["value"]
    write_to_influx_multiseries(df, None, tag_columns, field_columns, host, port, user, password, db_name, batch_size )


def parse_args():
    """
    Parse the args from main.
    """
    parser = argparse.ArgumentParser(
        description='example code to play with InfluxDB',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--debug', required=False, action='store_true',
                        help='Run in debug mode (turns on extra logging)')
    parser.add_argument('--host', type=str, required=False,
                        default='localhost',
                        help='hostname of InfluxDB http API')
    parser.add_argument('--dbname', type=str, required=False,
                        default='demo_db',
                        help='Influxdb database name')
    parser.add_argument('--port', type=int, required=False, default=8086,
                        help='port of InfluxDB http API')
    parser.add_argument('--user', type=str, required=False, default="admin",
                        help='Username for influxdb')
    parser.add_argument('--password', type=str, required=False, default=None,
                        help='Password for influxdb')
    parser.add_argument('--batch-size', type=int, required=False, default=10000,
                        help='Batch size towards influxdb')
    parser.add_argument('--taglist', type=str, required=False, default=None,
                        help='File with list of tags to download. If false do recursive downloads of all')
    parser.add_argument('--tagframe', type=str, required=False, default=None,
                        help='CSV of tags and type of measurment')
    parser.add_argument('--from-year', type=int, required=False, default=2015,
                        help='Lower bound of year to include from taglist')
    parser.add_argument('--to-year', type=int, required=False, default=2019,
                        help='Upper bound of year to include from taglist')
    parser.add_argument('--include', type=str, required=False, default=".*",
                        help='Regexp of files to include when doing recursive download')
    parser.add_argument('--exclude', type=str, required=False, default="a^",
                        help='Regexp of files to exclude when doing recursive download')
    parser.add_argument('--base-path', type=str, required=False,
                        default='/raw/corporate/Aspen%20MS%20-%20IP21%20Troll%20C/sensordata/1776-TROC',
                        help='Base path of either recursive search or tags')
    parser.add_argument('--wind-files', type=str, required=False,
                        default=None,
                        help='Directory with wind json-files')
    parser.add_argument('--iroc-lake', required=False, action='store_true',
                        help='Enter IROC crawling mode, special structure')
    parser.add_argument('--data_store_name', type=str, required=False,
                        default='dataplatformdlsprod',
                        help='Azure data store name')
    parser.add_argument('--para', type=int, required=False, default=16,
                        help='Level of parallelization to use')
    parser.add_argument('--dry-run', required=False, action='store_true',
                        help='Dry run? (Only show files which would have been imported, but don\'t do it)')
    parser.add_argument('--token-cache', type=str, required=False, default=None,
                        help='File to cache the azure token in. If it exists, use it as a token, '
                             'otherwise store token in it.')
    return parser.parse_args()


def extract_tags_from_df(data_frame):
    """
    Leftmost column should be well name, column names should be metric_type(e.g gassrate), and as elements the
    actual sensor names. Returns a dict from sensor names to a dictionary of its well-name and
    column-name (metric_type)
    """
    res = dict()
    for row in data_frame.itertuples():
        row_dict = row._asdict()
        del row_dict['Index']
        first_column_name, first_colum_val = row_dict.popitem(last=False)
        for measurement_name, tag_name in row_dict.items():
            res[tag_name] = {first_column_name: first_colum_val, "metric_type": measurement_name}
    return res


def main():
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    parallelization = args.para
    base_path = args.base_path
    dry_run = args.dry_run

    years = range(args.from_year, args.to_year)

    azure_data_store = args.data_store_name

    host = args.host
    port = args.port
    user = args.user
    password = args.password
    db_name = args.dbname
    batch_size = args.batch_size

    # Disable the Statoil proxy
    os.environ['NO_PROXY'] = host

    influx_settings = (host, port, user, password, db_name, batch_size)

    if not dry_run:
        client = DataFrameClient(host, port, user, password, db_name)
        client.create_database(db_name)
        # https://docs.influxdata.com/influxdb/v1.4/concepts/schema_and_data_layout/#shard-group-duration-management
        # In fact, increasing the shard group duration beyond the default seven day value can improve compression,
        # improve write speed, and decrease the fixed iterator overhead per shard group. Shard group durations of 50
        # years and over, for example, are acceptable configurations.
        client.query('ALTER RETENTION POLICY autogen ON {} SHARD DURATION 9000w'.format(db_name))

    # Should probably set maxtasksperchild = 10, but is only available in multiprocessing 2.7, and we have 2.6
    pool = multiprocessing.Pool(parallelization)

    before = timeit.default_timer()

    tag_map = dict() # A map from a Statoil tag (name of a measurement) to the influx-tags for it (e.g. well-name etc)

    if args.wind_files:  # local wind json files
        generator = walk_the_street(args.wind_files, args.include, args.exclude)
        for file_path, sensor_tags in generator:
            if dry_run:
                print("DRY RUN Tag {}, path {}".format(sensor_tags, file_path))
            else:
                pool.apply_async(read_and_push_wind_json,
                                 (file_path, sensor_tags, influx_settings,))

    elif args.iroc_lake:
        logger.debug("Crawling IROC Well data from {}".format(base_path))
        if args.token_cache:
            token = get_token(args.token_cache)
        else:
            token = lib.auth()
        generator = walk_azure(base_path, azure_data_store, token)
        for file_path in generator:
            if dry_run:
                print("DRY RUN path {}".format(file_path))
            else:
                pool.apply_async(azure_iroc_fetch_and_push_to_influx,
                                 (token, azure_data_store, file_path, influx_settings,))

    else:
        crawling = not (args.taglist or args.tagframe)  # Will we be crawling for tags, or read them from file
        live_run = not dry_run
        if live_run or crawling:
            if args.token_cache:
                token = get_token(args.token_cache)
            else:
                token = lib.auth()

        if args.taglist or args.tagframe:
            if args.taglist:
                logger.debug("Attempting to parse tags from the taglist %s" % args.taglist)
                with open(args.taglist) as f:
                    tag_list = f.readlines()
                    tag_list = [x.strip() for x in tag_list]
            else:
                logger.debug("Attempting to parse tags and metadata from the tagframe %s" % args.tagframe)
                df = pd.read_csv(args.tagframe, sep=";")
                tag_map = extract_tags_from_df(df)
                tag_list = [str(x).strip() for x in tag_map.keys()]
                if 'nan' in tag_list:
                    tag_list.remove('nan')
            generator = walk_tags(base_path, tag_list, years)

        else:  # Crawling IMS type structure
            generator = walk_and_tag_azure(base_path, azure_data_store, token, args.include, args.exclude)

        for file_path, tag in generator:
            sensor_tags = dict(tag_map.get(tag, dict()))
            sensor_tags.update({'tag': tag})
            if dry_run:
                print("DRY RUN Tag {}, path {}".format(sensor_tags, file_path))
            else:
                pool.apply_async(azure_fetch_and_push_to_influx,
                                 (token, azure_data_store, file_path, sensor_tags, influx_settings,))

    pool.close()
    pool.join()

    after = timeit.default_timer()
    print("Processing took %s seconds" % (after - before))


def get_token(token_cache):
    try:
        logger.debug("Attempting to open token cache %s" % token_cache)
        with open(token_cache, 'r') as f:
            token_dict = json.load(f)
            token = DataLakeCredential(token_dict)
            logger.debug("Token cache read from %s" % token_cache)
    except (IOError, ValueError):
        logger.debug("Error when opening token cache, creating new token")
        token = lib.auth()
        token_dict = token.token
        with open(token_cache, 'w') as f:
            json.dump(token_dict, f)
    return token


def read_and_push_wind_json(path, tags, influx_settings):
    df = read_wind_json_as_df(path)
    (host, port, user, password, db_name, batch_size) = influx_settings
    logger.debug("Attempting to send the wind-df from path %s" % path)
    write_to_influx(df, tags, host, port, user, password, db_name, batch_size, time_precision="ms")


def walk_tags(base_path, tag_list, years):
    for tag in tag_list:
        for year in years:
            file_path = base_path + "/{}/{}_{}.csv".format(tag, tag, year)
            yield file_path, tag

if __name__ == '__main__':
    main()
