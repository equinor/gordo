# -*- coding: utf-8 -*-

from typing import Optional, Union

import aiohttp
from werkzeug.exceptions import BadRequest


class HttpUnprocessableEntity(BaseException):
    """
    Represents an error from an HTTP status code of 422: UnprocessableEntity.
    Used in our case for calling /anomaly/prediction on a model which does not
    support anomaly behavior.
    """

    pass


async def get(
    url: str, *args, session: Optional[aiohttp.ClientSession] = None, **kwargs
) -> dict:
    """
    GET JSON form some endpoint

    Parameters
    ----------
    url: str
        Endpoint to make request to
    session: aiohttp.ClientSession
        Session to use for making the request

    Returns
    -------
    dict
        The JSON response from the endpoint
    """

    async def _get(session):
        async with session.get(url, *args, **kwargs) as resp:  # type: ignore
            return await _handle_response(resp)

    if session is None:  # We have to create a session which will be closed
        async with aiohttp.ClientSession() as session:
            return await _get(session)
    else:
        return await _get(session)


async def post(
    url: str, *args, session: Optional[aiohttp.ClientSession] = None, **kwargs
) -> dict:
    """
    POST JSON to some endpoint

    Parameters
    ----------
    url: str
        Endpoint to make request to
    session: aiohttp.ClientSession
        Session to use for making the request

    Returns
    -------
    dict
        The JSON response from the endpoint
    """

    async def _post(session):
        async with session.post(url, *args, **kwargs) as resp:
            return await _handle_response(resp)

    if session is None:  # We have to create a session which will be closed
        async with aiohttp.ClientSession() as session:
            return await _post(session)
    else:
        return await _post(session)


async def _handle_response(resp: aiohttp.ClientResponse) -> Union[dict, bytes]:
    if 200 <= resp.status <= 299:
        is_json = resp.content_type == "application/json"
        return await (resp.json() if is_json else resp.content.read())
    else:
        content = await resp.content.read()
        msg = f"Failed to get response with status code: {resp.status}: {content}"
        if resp.status == 422:
            raise HttpUnprocessableEntity()
        elif 400 <= resp.status <= 499:
            raise BadRequest(msg)
        else:
            raise IOError(msg)
