import torch
print(torch.cuda.is_available())  # Должно вернуть True
print(torch.cuda.current_device())  # Проверяет текущее устройство
print(torch.cuda.get_device_name(0))
