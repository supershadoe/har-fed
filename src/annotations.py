from enum import Enum

class TargetClass(Enum):
    OTHER = 0
    LYING = 1
    SITTING = 2
    STANDING = 3
    WALKING = 4
    RUNNING = 5
    CYCLING = 6
    NORDIC = 7
    WATCHING_TV = 9
    COMPUTER = 10
    DRIVING_CAR = 11
    ASCENDING = 12
    DESCENDING = 13
    VACUUM = 16
    IRONING = 17
    FOLDING = 18
    CLEANING = 19
    SOCCER = 20
    ROPE = 24

class Attributes:
    __slots__ = ['to_keep', 'to_drop']

    def __init__(self, *, is_preprocessing: bool) -> None:
        orient_labels = ('orient_x', 'orient_y', 'orient_z', 'orient_w')
        sensor_labels = [
            'temp_c',
            'acc16_x', 'acc16_y', 'acc16_z',
            'acc6_x', 'acc6_y', 'acc6_z',
            'gyr_x', 'gyr_y', 'gyr_z',
            'mgt_x', 'mgt_y', 'mgt_z',
        ]
        if is_preprocessing:
            sensor_labels.extend(orient_labels)
        self.to_keep = [
            'timestamp', 'activity_id', 'heart_rate',
            *(
                f'{i}_{j}'
                for i in ['hand', 'chest', 'ankle']
                for j in sensor_labels
            ),
        ]
        # Drop orientation labels as they are useless
        self.to_drop = [
            f'{i}_{j}'
            for i in ['hand', 'chest', 'ankle']
            for j in orient_labels
        ] if is_preprocessing else []
