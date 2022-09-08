import dictdiffer


def patch_dict(original_dict: dict, patch_dictionary: dict) -> dict:
    """Patches a dict with another. Patching means that any path defines in the
    patch is either added (if it does not exist), or replaces the existing value (if
    it exists). Nothing is removed from the original dict, only added/replaced.

    Parameters
    ----------
    original_dict : dict
        Base dictionary which will get paths added/changed
    patch_dictionary: dict
        Dictionary which will be overlaid on top of original_dict

    Examples
    --------
    >>> patch_dict({"highKey":{"lowkey1":1, "lowkey2":2}}, {"highKey":{"lowkey1":10}})
    {'highKey': {'lowkey1': 10, 'lowkey2': 2}}
    >>> patch_dict({"highKey":{"lowkey1":1, "lowkey2":2}}, {"highKey":{"lowkey3":3}})
    {'highKey': {'lowkey1': 1, 'lowkey2': 2, 'lowkey3': 3}}
    >>> patch_dict({"highKey":{"lowkey1":1, "lowkey2":2}}, {"highKey2":4})
    {'highKey': {'lowkey1': 1, 'lowkey2': 2}, 'highKey2': 4}

    Returns
    -------
    dict
        A new dictionary which is the result of overlaying `patch_dictionary` on top of
        `original_dict`

    """
    diff = dictdiffer.diff(original_dict, patch_dictionary)
    adds_and_mods = [(f, d, s) for (f, d, s) in diff if f != "remove"]
    return dictdiffer.patch(adds_and_mods, original_dict)
