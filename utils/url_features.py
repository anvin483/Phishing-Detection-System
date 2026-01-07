import re
import numpy as np
from urllib.parse import urlparse

SUSPICIOUS_KEYWORDS = [
    'login', 'secure', 'verify', 'account', 'update',
    'bank', 'paypal', 'signin', 'confirm'
]

SUSPICIOUS_TLDS = ['.xyz', '.top', '.tk', '.ml', '.ga']

def extract_url_features(url: str):
    url = url.lower()

    parsed = urlparse(url)

    features = [
        len(url),                          # URL length
        url.count('.'),                    # dots
        sum(c.isdigit() for c in url),     # digits
        int(parsed.scheme == 'https'),     # HTTPS
        int(any(k in url for k in SUSPICIOUS_KEYWORDS)),
        int(any(tld in url for tld in SUSPICIOUS_TLDS))
    ]

    return np.array(features).reshape(1, -1)
