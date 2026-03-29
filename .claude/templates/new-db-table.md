# Template: New Data Model / Storage

> В текущем MVP нет реляционной БД — используется in-memory хранилище.
> Этот шаблон описывает добавление нового Pydantic-домена в FastAPI
> и соответствующего TypeScript типа на фронте.

---

## Backend: новая Pydantic модель (schemas.py)

```python
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field
import uuid

# Статусы
MyItemStatus = Literal["active", "archived", "deleted"]

class MyItem(BaseModel):
    """Описание сущности."""
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    status: MyItemStatus = "active"
    value: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    meta: dict = Field(default_factory=dict)

class CreateMyItemRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    value: float = Field(default=0.0, ge=0.0)

class UpdateMyItemRequest(BaseModel):
    name: str | None = None
    value: float | None = None
    status: MyItemStatus | None = None
```

## Backend: In-Memory Store (services/my_item_service.py)

```python
from dataclasses import dataclass, field
from gomoku_api.models.schemas import MyItem, CreateMyItemRequest, UpdateMyItemRequest
from datetime import datetime, timezone

@dataclass
class MyItemService:
    _store: dict[str, MyItem] = field(default_factory=dict)

    def create(self, req: CreateMyItemRequest) -> MyItem:
        item = MyItem(name=req.name, value=req.value)
        self._store[item.item_id] = item
        return item

    def get(self, item_id: str) -> MyItem | None:
        return self._store.get(item_id)

    def list_all(self) -> list[MyItem]:
        return list(self._store.values())

    def update(self, item_id: str, req: UpdateMyItemRequest) -> MyItem | None:
        item = self._store.get(item_id)
        if not item:
            return None
        if req.name is not None:
            item.name = req.name
        if req.value is not None:
            item.value = req.value
        if req.status is not None:
            item.status = req.status
        item.updated_at = datetime.now(timezone.utc)
        return item

    def delete(self, item_id: str) -> bool:
        return self._store.pop(item_id, None) is not None
```

## Backend: Router (routers/my_items.py)

```python
from fastapi import APIRouter, Request, HTTPException
from gomoku_api.models.schemas import MyItem, CreateMyItemRequest, UpdateMyItemRequest

router = APIRouter(prefix="/items", tags=["items"])

@router.post("", response_model=MyItem, status_code=201)
async def create_item(body: CreateMyItemRequest, request: Request) -> MyItem:
    return request.app.state.my_items.create(body)

@router.get("/{item_id}", response_model=MyItem)
async def get_item(item_id: str, request: Request) -> MyItem:
    item = request.app.state.my_items.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@router.get("", response_model=list[MyItem])
async def list_items(request: Request) -> list[MyItem]:
    return request.app.state.my_items.list_all()

@router.patch("/{item_id}", response_model=MyItem)
async def update_item(item_id: str, body: UpdateMyItemRequest, request: Request) -> MyItem:
    item = request.app.state.my_items.update(item_id, body)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item
```

## Frontend: TypeScript тип (api/types.ts)

```ts
export type MyItemStatus = "active" | "archived" | "deleted";

export interface MyItem {
  itemId: string;
  name: string;
  status: MyItemStatus;
  value: number;
  createdAt: string;   // ISO8601
  updatedAt?: string;
  meta?: Record<string, unknown>;
}

export interface CreateMyItemRequest {
  name: string;
  value?: number;
}
```

## Frontend: API client (api/client.ts)

```ts
export async function createItem(req: CreateMyItemRequest): Promise<MyItem> {
  const res = await fetch(`${BASE}/items`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

export async function getItem(itemId: string): Promise<MyItem> {
  const res = await fetch(`${BASE}/items/${itemId}`);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}
```

## Checklist

- [ ] Pydantic модель в `schemas.py`
- [ ] Service класс в `services/`
- [ ] Router в `routers/`
- [ ] Зарегистрировать в `main.py` (lifespan + include_router)
- [ ] TypeScript тип в `api/types.ts`
- [ ] API функции в `api/client.ts`
- [ ] Тесты (CRUD happy path + 404)

## Когда нужна настоящая БД

Если данные должны переживать перезапуск сервиса:
- Подключить SQLite через `aiosqlite` + `SQLAlchemy async`
- Или использовать простое JSON-файловое хранилище через `aiofiles`
- Миграции через `alembic`
