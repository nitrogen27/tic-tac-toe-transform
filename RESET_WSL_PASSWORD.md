# Сброс пароля WSL Ubuntu

## Способ 1: Через root пользователя (самый простой)

### Шаг 1: Запустите WSL от имени root

В PowerShell выполните:

```powershell
wsl -u root
```

Или для конкретного дистрибутива:

```powershell
wsl -d Ubuntu20.04LTS -u root
```

Или:

```powershell
wsl -d Ubuntu -u root
```

### Шаг 2: Сбросьте пароль

После входа как root выполните:

```bash
passwd nitrogen
```

Введите новый пароль дважды (пароль не будет отображаться при вводе).

### Шаг 3: Выйдите и войдите снова

```bash
exit
```

Затем войдите обычным способом:

```bash
wsl
```

## Способ 2: Если root не доступен

### Шаг 1: Остановите WSL

```powershell
wsl --shutdown
```

### Шаг 2: Запустите WSL от имени root через cmd

```powershell
# В PowerShell
wsl -u root

# Или в cmd
wsl -u root
```

### Шаг 3: Сбросьте пароль

```bash
passwd nitrogen
```

## Способ 3: Через настройку default user

Если root все еще не работает, можно изменить пользователя по умолчанию:

```powershell
# В PowerShell
ubuntu config --default-user root

# Затем запустите
wsl

# Сбросьте пароль
passwd nitrogen

# Верните обычного пользователя
ubuntu config --default-user nitrogen
```

## Способ 4: Установка нового пароля через командную строку

```powershell
# В PowerShell
wsl -u root -e bash -c "echo 'nitrogen:новый_пароль' | chpasswd"
```

**ВНИМАНИЕ:** Замените `новый_пароль` на желаемый пароль!

## Проверка

После сброса проверьте:

```bash
wsl
# Введите новый пароль когда попросит
```

## Если ничего не помогло

Можно переустановить WSL дистрибутив (данные сохранятся):

```powershell
wsl --unregister Ubuntu20.04LTS
# Затем переустановите из Microsoft Store
```

Или создайте нового пользователя с sudo правами.



