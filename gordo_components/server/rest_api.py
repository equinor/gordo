# -*- coding: utf-8 -*-

from flask import url_for
from flask_restplus import Api as BaseApi


class Api(BaseApi):
    """
    Redefine Api to ensure specs url is relative
    """

    @property
    def specs_url(self):
        return url_for(self.endpoint("specs"), _external=False)
