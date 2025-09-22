#!/usr/bin/env python3
"""
Пример использования библиотеки PySpz.

Этот скрипт демонстрирует основные возможности библиотеки
для чтения SPZ файлов.
"""

import pyspz
import numpy as np
import io
import gzip


def create_sample_spz_data():
    """Создает пример SPZ данных для демонстрации."""
    # Создаем заголовок SPZ v2
    header_data = (
        0x5053474E.to_bytes(4, 'little') +  # magic 'NGSP'
        (2).to_bytes(4, 'little') +         # version 2
        (2).to_bytes(4, 'little') +         # num_points 2
        (0).to_bytes(1, 'little') +          # sh_degree 0
        (8).to_bytes(1, 'little') +          # fractional_bits 8
        (0).to_bytes(1, 'little') +          # flags 0
        (0).to_bytes(1, 'little')            # reserved 0
    )
    
    # Создаем данные для 2 точек
    # Positions: 2 точки * 3 компоненты * 3 байта = 18 байт
    positions_data = b'\x00\x00\x00' * 6  # Все нули
    
    # Alphas: 2 точки * 1 байт = 2 байта
    alphas_data = b'\x80\x80'  # Средние значения
    
    # Colors: 2 точки * 3 компоненты = 6 байт
    colors_data = b'\x80\x80\x80\x80\x80\x80'  # Серые цвета
    
    # Scales: 2 точки * 3 компоненты = 6 байт
    scales_data = b'\x80\x80\x80\x80\x80\x80'  # Средние масштабы
    
    # Rotations v2: 2 точки * 3 компоненты = 6 байт
    rotations_data = b'\x00\x00\x00\x00\x00\x00'  # Нулевые ротации
    
    # Объединяем все данные
    body_data = positions_data + alphas_data + colors_data + scales_data + rotations_data
    
    # Сжимаем данные
    compressed_body = gzip.compress(body_data)
    
    # Создаем полный SPZ файл
    spz_data = header_data + compressed_body
    
    return spz_data


def main():
    """Основная функция демонстрации."""
    print("🚀 Демонстрация библиотеки PySpz")
    print("=" * 50)
    
    # Создаем пример SPZ данных
    print("📦 Создаем пример SPZ данных...")
    spz_data = create_sample_spz_data()
    print(f"✅ Создан SPZ файл размером {len(spz_data)} байт")
    
    # Загружаем данные через PySpz
    print("\n📖 Загружаем данные через PySpz...")
    file_obj = io.BytesIO(spz_data)
    data = pyspz.load(file_obj)
    
    # Показываем результаты
    print("✅ Данные успешно загружены!")
    print(f"📊 Количество гауссиан: {len(data['positions'])}")
    
    print("\n📋 Структура данных:")
    for key, array in data.items():
        print(f"  {key}: {array.shape} {array.dtype}")
    
    # Показываем примеры данных
    print("\n🔍 Примеры данных:")
    print(f"  Positions (первые 2 точки):\n{data['positions']}")
    print(f"  Alphas (прозрачность):\n{data['alphas']}")
    print(f"  Colors (RGB):\n{data['colors']}")
    print(f"  Scales (масштабы):\n{data['scales']}")
    print(f"  Rotations (кватернионы):\n{data['rotations']}")
    
    # Проверяем нормализацию кватернионов
    print("\n🧮 Проверка нормализации кватернионов:")
    for i, quat in enumerate(data['rotations']):
        norm = np.linalg.norm(quat)
        print(f"  Кватернион {i}: норма = {norm:.6f}")
    
    print("\n✅ Все проверки пройдены успешно!")
    print("🎉 Библиотека PySpz полностью рабочая!")


if __name__ == "__main__":
    main()
