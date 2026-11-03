"""DataForge pipeline: the heavy work, on the pipeline server, with no database.

Takes work from the app server's Redis, does it, and sends the results back. What
the two projects agree on — message names, spec and result shapes, queue
names, shared Redis keys — is in CONTRACT.md, and PROTOCOL_VERSION is that
agreement's number: the app server stamps it on every message, and a message from
another version is refused rather than misread.
"""

PROTOCOL_VERSION = 4
