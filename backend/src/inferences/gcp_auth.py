"""
Cheap, offline check for whether Google Cloud credentials are configured.

Why this exists
---------------
Constructing a Google client (LanguageServiceClient, aiplatform.init, ...)
without credentials does not fail fast. Application Default Credentials falls
back to probing the GCE metadata server at 169.254.169.254, which does not
exist outside Google Cloud, so the call blocks until the network times out —
tens of seconds on a public deployment.

Checking for credentials first turns that hang into an instant, deliberate
fallback to the offline path.
"""

import os
from pathlib import Path


def adc_available() -> bool:
    """
    True if Application Default Credentials are plausibly configured.

    Covers both ways credentials are normally supplied:
      1. GOOGLE_APPLICATION_CREDENTIALS pointing at a service-account key file.
      2. `gcloud auth application-default login`, which writes a credentials
         file to a well-known per-user location.

    This is intentionally a availability check, not a validity check: it never
    performs a network call. An invalid key still fails later, and callers are
    expected to handle that.
    """
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if key_path and Path(key_path).is_file():
        return True

    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if not appdata:
            return False
        well_known = Path(appdata) / "gcloud" / "application_default_credentials.json"
    else:
        well_known = (
            Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        )

    return well_known.is_file()
