from aiohttp.web_urldispatcher import View
from aiohttp_cors import CorsViewMixin


class BaseView(View, CorsViewMixin):
    URL_PATH: str

    @property
    def cache(self):
        return self.request.app['cache']

    def w2v(self):
        return self.request.app['w2v']
