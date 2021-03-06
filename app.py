import logging

from types import AsyncGeneratorType, MappingProxyType
from typing import AsyncIterable, Mapping

from aiohttp import PAYLOAD_REGISTRY
from aiohttp.web_app import Application
from aiohttp_apispec import setup_aiohttp_apispec, validation_middleware
import aiohttp_cors

from handlers import HANDLERS
from payloads import AsyncGenJSONListPayload, JsonPayload
from plugins.w2v import FastTextVectorizer
from plugins.config import cfg

api_address = "0.0.0.0"
api_port = 8081

MEGABYTE = 1024 ** 2
MAX_REQUEST_SIZE = 70 * MEGABYTE

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)


def create_app() -> Application:
    """
    Создает экземпляр приложения, готового к запуску
    """
    # TODO добавить middlewares для вадидации полей сообщений
    app = Application(
        client_max_size=MAX_REQUEST_SIZE,
        # middlewares=[check_token_middleware]
    )
    aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    # Регистрация обработчика
    for handler in HANDLERS:
        log.debug('Registering handler %r as %r', handler, handler.URL_PATH)

        route = app.router.add_route('*', handler.URL_PATH, handler)

        app['aiohttp_cors'].add(route)

    # регестрируем w2v
    log.debug('Registering w2v')
    app['w2v'] = FastTextVectorizer(cfg.app.directory.model_path)

    setup_aiohttp_apispec(app=app, title="I SEE YOU API", swagger_path='/')

    # Автоматическая сериализация в json данных в HTTP ответах
    PAYLOAD_REGISTRY.register(AsyncGenJSONListPayload,
                              (AsyncGeneratorType, AsyncIterable))
    PAYLOAD_REGISTRY.register(JsonPayload, (Mapping, MappingProxyType))
    return app
