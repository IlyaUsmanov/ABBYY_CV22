# Демозаикинг

1-ое ДЗ по курсу CV

## Результаты работы алгоритмов

### Восстановленные изображение

В папке [results](https://github.com/IlyaUsmanov/ABBYY_CV22_Demosaicing/tree/main/results)

### PSNR и время работы

**Biliniear interpolation**

* PSNR: -2.9914229579467078

* Time elapsed per 1 MP: 0.6456190215948232


**Improved interpolation**

Оригинальная [статья](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1326587&casa_token=TG3mYuZt1VwAAAAA:wlvAKOvwXRlbF74gcgcqA6_pcJpFSAgdQlYLgH9U0lXbpI9WUZ9dW-FRS5sMiwLFjgAK4Ao9i9k&tag=1)

* PSNR: 1.8584490275559498

* Time elapsed per 1 MP: 0.422766802376174


**VNG**

* PSNR: 1.4531627459080767

* Time elapsed per 1 MP: 2.3404172430486105


## Выводы

Улучшенная билинейная интерполяция показала себя лучше, чем VNG как по скорости, так и по качеству, однако оба метода одинаково плохо справляеются со скоплениями черных точек(превращают их в радужные) и границами цветных объектов