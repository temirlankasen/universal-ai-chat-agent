import decimal
from datetime import datetime

import numpy as np
import orjson


def json_dumps(raw_json: dict | list) -> str:
    def unstandart_encoder(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    return orjson.dumps(raw_json, default=unstandart_encoder).decode("utf-8")