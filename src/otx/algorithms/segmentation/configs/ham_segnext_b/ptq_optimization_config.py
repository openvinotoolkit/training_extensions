"""PTQ config file."""
from nncf import IgnoredScope

ignored_scope = IgnoredScope(
    patterns=["/hamburger/"],
    types=[
        "Add",
        "MVN",
        "Divide",
        "Multiply",
    ],
)
