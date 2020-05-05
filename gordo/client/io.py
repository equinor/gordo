# -*- coding: utf-8 -*-

from typing import Union, Optional

import requests


class HttpUnprocessableEntity(Exception):
    """
    Represents an error from an HTTP status code of 422: UnprocessableEntity.
    Used in our case for calling /anomaly/prediction on a model which does not
    support anomaly behavior.
    """

    pass


class ResourceGone(Exception):
    """
    Represents an error from an HTTP status code of 410: Gone.
    Indicates that access to the target resource is no longer available at the origin
    server and that this condition is likely to be permanent.

    Used in our case for calling the server with a revision which is no longer used.
    """

    pass


class BadGordoRequest(Exception):
    """
    Represents a general 4xx bad request
    """

    pass


class NotFound(Exception):
    """
    Represents a 404
    """

    pass


def _handle_response(
    resp: requests.Response, resource_name: Optional[str] = None
) -> Union[dict, bytes]:
    """
    Handles the response from the server by either returning the parsed json
    (if it is json), the pure bytestream of the content, or raise an exception
    if something went wrong.


    Parameters
    ----------
    resp:
        The request to inspect for a result
    resource_name:
        An optional name to add to error messages. Should describe the resource we
        attempted to GET

    Returns
    -------
     Union[dict, bytes]

    Raises
    ------
    HttpUnprocessableEntitys
        In case of a 422 from the server
    ResourceGone
        In case of a 410 from the server
    NotFound
        In case of a 404 from the server
    BadGordoRequest
        Any other 4xx error
    IOError
        In case of network or IO errors
    """
    if 200 <= resp.status_code <= 299:
        is_json = resp.headers["content-type"] == "application/json"
        return resp.json() if is_json else resp.content
    else:
        if resource_name:
            msg = (
                f"We failed to get response while fetching resource: {resource_name}. "
                f"Return code: {resp.status_code}. Return content: {resp.content!r}"
            )
        else:
            msg = f"Failed to get response: {resp.status_code}: {resp.content!r}"

        if resp.status_code == 422:
            raise HttpUnprocessableEntity(msg)
        elif resp.status_code == 410:
            raise ResourceGone(msg)
        elif resp.status_code == 404:
            raise NotFound(msg)
        elif 400 <= resp.status_code <= 499:
            raise BadGordoRequest(msg)
        else:
            raise IOError(msg)
