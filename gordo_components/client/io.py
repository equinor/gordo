# -*- coding: utf-8 -*-

from typing import Optional

import aiohttp
from werkzeug.exceptions import BadRequest


async def fetch_json(
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

    async def fetch(session):
        async with session.get(url, *args, **kwargs) as resp:  # type: ignore
            return await _handle_json(resp)

    if session is None:  # We have to create a session which will be closed
        async with aiohttp.ClientSession() as session:
            return await fetch(session)
    else:
        return await fetch(session)


async def post_json(
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

    async def post(session):
        async with session.post(url, *args, **kwargs) as resp:
            return await _handle_json(resp)

    if session is None:  # We have to create a session which will be closed
        async with aiohttp.ClientSession() as session:
            return await post(session)
    else:
        return await post(session)


async def _handle_json(resp: aiohttp.ClientResponse) -> dict:
    if 200 <= resp.status <= 299:
        return await resp.json()
    else:
        content = await resp.content.read()
        msg = f"Failed to get JSON with status code: {resp.status}: {content}"

        if 400 <= resp.status <= 499:
            raise BadRequest(msg)
        else:
            raise IOError(msg)
