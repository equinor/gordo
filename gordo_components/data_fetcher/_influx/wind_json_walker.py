# -*- coding: utf-8 -*-

import os
import pandas as pd
import re
import multiprocessing, logging


logger = multiprocessing.log_to_stderr()  # TODO: This cant be right
logger.setLevel(logging.INFO)


def read_wind_json_as_df(path):
    logger.debug("Attempting to parse the wind-json %s" % path)
    with open(path) as f:
        df = pd.read_json(f, convert_dates=["t"])
        df.rename(index=str, columns={"t": "Timestamp", "v": "value"}, inplace=True)
        df = df.set_index("Timestamp")
    return df


def walk_the_street(path, include_regexp=".*json", exclude_regexp="a^"):
    """Generator of all files from a given path matching include_regexp and not matching exclude_regexp. Assumes the
    matching files to have the windmill-format: Windfarm-Windmill-Measurement_TS_DATE, and will attempt to to parse
    all matching files according to this. Returns paths of files and a map with tags generated from the file."""
    include_regexp = re.compile(include_regexp)
    exclude_regexp = re.compile(exclude_regexp)
    logger.debug("Starting recursive traversal from %s" % path)
    for dirpath, dirnames, files in os.walk(path):
        for file_path in sorted(files):
            if include_regexp.match(file_path) and not exclude_regexp.match(file_path):
                logger.debug("Returning file %s" % file_path)
                pre = file_path.split("_TS_")[0]
                components = pre.split("-")
                tags = {
                    "Windfarm": components[0],
                    "Windmill": components[1],
                    "Measurement": components[2],
                }
                yield os.path.join(dirpath, file_path), tags
            else:
                logger.debug(
                    "Found that the file_path %s does not satisfy regexp requirements"
                    % file_path
                )
