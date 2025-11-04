# Запуск проекта через Docker

## Предварительные требования

1. **Docker Desktop** должен быть установлен и запущен
   - Скачать: https://www.docker.com/products/docker-desktop
   - Для Windows: включите WSL2 backend в настройках

2. **Проверка установки**:
   ```bash
   docker --version
   docker-compose --version
   ```

## Быстрый старт

### 1. Сборка и запуск всех сервисов

```bash
docker-compose up --build
```

Это команда:
- Соберет образы для server и client
- Запустит оба контейнера
- Сервер будет доступен на `ws://localhost:8080`
- Клиент будет доступен на `http://localhost:5173`

### 2. Запуск в фоновом режиме

```bash
docker-compose up -d --build
```

### 3. Просмотр логов

```bash
# Все сервисы
docker-compose logs -f

# Только сервер
docker-compose logs -f server

# Только клиент
docker-compose logs -f client
```

### 4. Остановка

```bash
docker-compose down
```

### 5. Остановка с удалением volumes

```bash
docker-compose down -v
```

## Отдельный запуск сервисов

### Только сервер

```bash
# Сборка образа
docker build -f Dockerfile.server -t tic-tac-toe-server .

# Запуск контейнера
docker run -d \
  --name tic-tac-toe-server \
  -p 8080:8080 \
  -v ./server/saved:/app/server/saved \
  tic-tac-toe-server
```

### Только клиент

```bash
# Сборка образа
docker build -f Dockerfile.client -t tic-tac-toe-client .

# Запуск контейнера
docker run -d \
  --name tic-tac-toe-client \
  -p 5173:5173 \
  -v ./client:/app/client \
  -v /app/client/node_modules \
  tic-tac-toe-client
```

## Решение проблем

### Пересборка образов

```bash
docker-compose build --no-cache
```

### Очистка Docker

```bash
# Удаление остановленных контейнеров
docker container prune

# Удаление неиспользуемых образов
docker image prune

# Полная очистка (осторожно!)
docker system prune -a
```

### Проверка работающих контейнеров

```bash
docker ps
```

### Вход в контейнер

```bash
# В контейнер сервера
docker exec -it tic-tac-toe-server bash

# В контейнер клиента
docker exec -it tic-tac-toe-client bash
```

## Особенности

- **Модели сохраняются** в `./server/saved` директории на хосте
- **Hot-reload для клиента** работает благодаря volume mount
- **TensorFlow.js** устанавливается автоматически в Linux окружении
- **GPU поддержка** (CUDA) доступна если Docker имеет доступ к GPU (требует nvidia-docker)

## GPU поддержка (опционально)

Если у вас есть NVIDIA GPU и установлен nvidia-docker:

```yaml
# Добавьте в docker-compose.yml в секцию server:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Полезные команды

```bash
# Перезапуск сервисов
docker-compose restart

# Обновление и пересборка
docker-compose up -d --build --force-recreate

# Просмотр использования ресурсов
docker stats
```



