class Annotations:
    __slots__ = ['attrs', 'attrs_to_drop']
    CLASS_NAMES = {
        0: 'other',
        1: 'lying',
        2: 'sitting',
        3: 'standing',
        4: 'walking',
        5: 'running',
        6: 'cycling',
        7: 'Nordic',
        9: 'watching',
        10: 'computer',
        11: 'car',
        12: 'ascending',
        13: 'descending',
        16: 'vacuum',
        17: 'ironing',
        18: 'folding',
        19: 'house',
        20: 'playing',
        24: 'rope',
    }

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
        self.attrs = [
            'timestamp', 'activity_id', 'heart_rate',
            *(
                f'{i}_{j}'
                for i in ['hand', 'chest', 'ankle']
                for j in sensor_labels
            ),
        ]
        # Drop orientation labels as they are useless
        self.attrs_to_drop = [
            f'{i}_{j}'
            for i in ['hand', 'chest', 'ankle']
            for j in orient_labels
        ] if is_preprocessing else []
