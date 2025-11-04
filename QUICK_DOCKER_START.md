# Быстрый старт с Docker

## Шаг 1: Запустите Docker Desktop

Убедитесь, что **Docker Desktop запущен**:
- Проверьте иконку Docker в системном трее (Windows)
- Если Docker не запущен, откройте Docker Desktop из меню Пуск

## Шаг 2: Проверьте, что Docker работает

```bash
docker ps
```

Должна выполниться без ошибок.

## Шаг 3: Запустите проект

```bash
docker-compose up --build
```

Эта команда:
- ✅ Соберет образы для server и client
- ✅ Установит все зависимости (включая TensorFlow.js)
- ✅ Запустит сервер на `ws://localhost:8080`
- ✅ Запустит клиент на `http://localhost:5173`

## Шаг 4: Откройте приложение

Откройте в браузере: **http://localhost:5173**

## Полезные команды

```bash
# Запуск в фоновом режиме
docker-compose up -d --build

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down

# Перезапуск
docker-compose restart
```

## Если возникли проблемы

1. **Docker daemon не запущен**: Запустите Docker Desktop
2. **Порты заняты**: Измените порты в `docker-compose.yml`
3. **Ошибки сборки**: Попробуйте `docker-compose build --no-cache`



