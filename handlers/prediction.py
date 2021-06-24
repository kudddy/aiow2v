import logging

from aiohttp.web_response import Response
from aiohttp_apispec import docs, response_schema

from .base import BaseView

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

log.setLevel(logging.INFO)


class PredictionHandler(BaseView):
    URL_PATH = r'/getnearesttoken/{token}/{n}'

    @property
    def n(self):
        return int(self.request.match_info.get('n'))

    @property
    def token(self):
        return str(self.request.match_info.get('token'))

    @docs(summary="Возвращает n число ближайших соседей токена", tags=["Basic methods"],
          description="Ручка для расширения поисковых индексов на основе w2v",
          )
    # @response_schema(description="Возвращает n число ближайших соседей токена"
    #                              "сортированные по дате")
    async def post(self):
        status: bool = True
        result: list = []
        try:
            logging.info("message_name - %r info - %r", "GET_NEAREST_TOKEN", "token - {}".format(self.token))
            result: list = self.w2v().wv.most_similar([self.token], topn=self.n)
            return Response(body={"MESSAGE_NAME": "GET_NEAREST_TOKEN",
                                  "STATUS": status,
                                  "PAYLOAD": {
                                      "result": result,
                                      "description": "OK"
                                  }})
        except Exception as e:
            logging.info("message_name - %r info - %r error - %r",
                         "GET_NEAREST_TOKEN",
                         "token - {}".format(self.token),
                         e)
            status = False
            return Response(body={"MESSAGE_NAME": "GET_NEAREST_TOKEN",
                                  "STATUS": status,
                                  "PAYLOAD": {
                                      "result": result,
                                      "description": "error"
                                  }})
