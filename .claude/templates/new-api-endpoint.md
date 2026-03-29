# Template: New API Endpoint

## Шаги

1. Добавить Pydantic схемы в `apps/api/src/gomoku_api/models/schemas.py`
2. Добавить роут в нужный router (`routers/engine.py` или `routers/training.py`)
3. Добавить бизнес-логику в сервис (`services/`)
4. Добавить тест в `tests/`
5. Обновить `apps/api/README.md` (таблица эндпоинтов)

---

## Pydantic схемы (schemas.py)

```python
class MyRequest(BaseModel):
    position: Position
    my_param: int = Field(default=5, ge=1, le=20)

class MyResponse(BaseModel):
    result: str
    value: float
    meta: dict[str, Any] = Field(default_factory=dict)
```

## Router (routers/my_router.py)

```python
from fastapi import APIRouter, Request
from gomoku_api.models.schemas import MyRequest, MyResponse

router = APIRouter(tags=["my-tag"])

@router.post("/my-endpoint", response_model=MyResponse)
async def my_endpoint(body: MyRequest, request: Request) -> MyResponse:
    """Описание эндпоинта."""
    service = request.app.state.my_service
    return await service.do_something(body)
```

## Регистрация в main.py

```python
from gomoku_api.routers import my_router

app.include_router(my_router)
```

## Сервис (services/my_service.py)

```python
from dataclasses import dataclass
from gomoku_api.models.schemas import MyRequest, MyResponse

@dataclass
class MyService:
    async def do_something(self, req: MyRequest) -> MyResponse:
        # логика здесь
        return MyResponse(result="ok", value=0.0)
```

## Инициализация в lifespan (main.py)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = EngineAdapter()
    app.state.my_service = MyService()   # добавить здесь
    yield
```

## Тест (tests/test_my_router.py)

```python
import pytest
from httpx import AsyncClient, ASGITransport
from gomoku_api.main import app

@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

@pytest.mark.asyncio
async def test_my_endpoint_ok(client):
    payload = {"position": {...}, "my_param": 3}
    resp = await client.post("/my-endpoint", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data

@pytest.mark.asyncio
async def test_my_endpoint_validation(client):
    resp = await client.post("/my-endpoint", json={"my_param": 99})  # без position
    assert resp.status_code == 422
```

## Checklist

- [ ] Схемы в `schemas.py`
- [ ] Роут в router файле
- [ ] Сервис в `services/`
- [ ] Регистрация в `main.py`
- [ ] Тест (happy path + validation error)
- [ ] Обновить `apps/api/README.md`
