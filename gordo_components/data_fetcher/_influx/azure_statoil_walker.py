# -*- coding: utf-8 -*-

import collections

import azure.datalake
import numpy as np
import os
import pandas as pd
import re
from azure.datalake.store import core
import multiprocessing, logging

logger = multiprocessing.log_to_stderr()  # TODO: This cant be right
logger.setLevel(logging.INFO)


def read_df_from_azure(azure_data_store, file_path, token):
    """ 
    Reads the file in file_path from the azure datalake azure_data_store using access token token,
    attempts to read it as a csv into panda. Expects the csv to have 4 columns of types String, Number, Timestamp,
    and Number. Only the timestamp and number is used, in a timestamp -> number mapping. The resulting DataFrame
    is returned
    """
    adls_file_system_client = core.AzureDLFileSystem(token, store_name=azure_data_store)
    try:
        logger.debug(
            "Attempting to open file {} on azure_data_store {}".format(
                file_path, azure_data_store
            )
        )
        with adls_file_system_client.open(file_path, "rb") as f:
            print("Parsing file {}".format(file_path))
            df = pd.read_csv(
                f,
                sep=";",
                header=None,
                names=["Sensor", "Value", "Timestamp", "Status"],
                dtype={"Value": np.float64},
            )
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df = df.set_index("Timestamp")
            df = df.drop(["Sensor", "Status"], axis=1)
            print(df.head(10))
    except azure.datalake.store.exceptions.FileNotFoundError:
        print("Azure File not found: %s" % file_path)
        logger.warning("Azure File not found: %s" % file_path)
        df = None
    return df


def read_iroc_df_from_azure(azure_data_store, file_path, token):
    """ 
    Reads the file in file_path from the azure datalake azure_data_store using access token token,
    attempts to read it as a csv into pandas.
    """
    adls_file_system_client = core.AzureDLFileSystem(token, store_name=azure_data_store)
    try:
        logger.debug(
            "Attempting to open IROC file {} on {}".format(file_path, azure_data_store)
        )
        with adls_file_system_client.open(file_path, "rb") as f:
            print("Parsing file {}".format(file_path))
            df = pd.read_csv(f, sep=",")
            # Note, there are some "digital" sensors with string values, now they are just NaN converted
            df["value"] = df["value"].apply(pd.to_numeric, errors="coerce")
            df.dropna(inplace=True, subset=["value"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            df["site"], df["service-well"], df["sensor"] = df.tag.str.split(".").str
            df["service"], df["well"] = df["service-well"].str.split("::").str
            df = df.drop(["service-well"], axis=1)
            print(df.head(10))
    except azure.datalake.store.exceptions.FileNotFoundError:
        print("Azure File not found: %s" % file_path)
        logger.warning("Azure File not found: %s" % file_path)
        df = None
    return df


def walk_and_tag_azure(base_path, 
                       azure_data_store, 
                       token, 
                       include_regexp=".*", 
                       exclude_regexp="a^"
    ):
    """
    Uses walk_azure to generate paths according to its documentation, but also parses the resulting
    file-names for a Statoil tag-name, and returns that in addition to the file-path
    """
    for file_path in walk_azure(
        base_path, azure_data_store, token, include_regexp, exclude_regexp
    ):
        tag = os.path.basename(os.path.dirname(file_path))
        yield file_path, tag


def walk_azure(base_path, 
               azure_data_store, 
               token, 
               include_regexp=".*", 
               exclude_regexp="a^"):
    """Walks azure datalake "azure_data_store" in a depth-first fashion from base_path, yielding files
      which match include_re AND not match exclude_re"""
    logger.debug(
        "Starting a azure walk with base_path: {} on data_store {} with include_regexp: {} "
        "and exclude_regexp: {}".format(
            base_path, azure_data_store, include_regexp, exclude_regexp
        )
    )

    adls_file_system_client = core.AzureDLFileSystem(token, store_name=azure_data_store)
    fringe = collections.deque(adls_file_system_client.ls(base_path, detail=True))
    include_regexp = re.compile(include_regexp)
    exclude_regexp = re.compile(exclude_regexp)
    while fringe:
        a_path = fringe.pop()
        if a_path["type"] == "DIRECTORY":
            logger.debug("Expanding the directory %s" % a_path["name"])
            fringe.extend(adls_file_system_client.ls(a_path["name"], detail=True))
        if a_path["type"] == "FILE":
            file_path = a_path["name"]
            if include_regexp.match(file_path) and not exclude_regexp.match(file_path):
                logger.debug("Returning the file_path %s" % file_path)
                yield file_path
            else:
                logger.debug(
                    "Found that the file_path %s does not satisfy regexp requirements"
                    % file_path
                )
