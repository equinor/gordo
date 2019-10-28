# -*- coding: utf-8 -*-

from typing import Union

import requests
from werkzeug.exceptions import BadRequest


class HttpUnprocessableEntity(BaseException):
    """
    Represents an error from an HTTP status code of 422: UnprocessableEntity.
    Used in our case for calling /anomaly/prediction on a model which does not
    support anomaly behavior.
    """

    pass


def _handle_response(resp: requests.Response) -> Union[dict, bytes]:
    if 200 <= resp.status_code <= 299:
        is_json = resp.headers["content-type"] == "application/json"
        return resp.json() if is_json else resp.content
    else:
        msg = f"Failed to get response: {resp.status_code}: {resp.content!r}"
        if resp.status_code == 422:
            raise HttpUnprocessableEntity()
        elif 400 <= resp.status_code <= 499:
            raise BadRequest(msg)
        else:
            raise IOError(msg)
