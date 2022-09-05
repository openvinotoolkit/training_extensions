"""This file defines the URL class."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from typing import Union


class URL:
    """Class for working with URLs.

    Provides functions for getting components of the URL such as <scheme>://<host>/<path>
    """

    def __parse(self, url):
        """Parse URL into 3 internal saved components.

        (Function stolen from urllib library)
        <scheme>://<netloc>/<path>
        """

        def __splitnetloc(url, start=0):
            delimiter = len(url)  # position of end of domain part of url, default is end
            for char in "/?#":  # look for delimiters; the order is NOT important
                position = url.find(char, start)  # find first of this delimiter
                if position >= 0:  # if found
                    delimiter = min(delimiter, position)  # use earliest delimiter position
            return url[start:delimiter], url[delimiter:]  # return (domain, rest)

        if url is None:
            raise ValueError("Cannot build an URL from None")

        scheme_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-."
        netloc = scheme = ""
        i = url.find(":")

        if i > 0:
            for character in url[:i]:
                if character not in scheme_chars:
                    break
            else:
                rest = url[i + 1 :]  # make sure "url" is not actually a port number (in which case
                if not rest or any(c not in "0123456789" for c in rest):
                    scheme, url = url[:i].lower(), rest  # Not a port

        if url[:2] == "//":
            netloc, url = __splitnetloc(url, 2)
            if ("[" in netloc and "]" not in netloc) or ("]" in netloc and "[" not in netloc):
                raise ValueError("Invalid IPv6 URL")

        if "#" in url:
            url, _ = url.split("#", 1)
        if "?" in url:
            url, _ = url.split("?", 1)

        self.__scheme = scheme
        self.__netloc = netloc
        self.__path = url

    def __init__(self, url: Union["URL", str]):
        self.__url = str(url)
        self.__scheme = ""
        self.__netloc = ""
        self.__path = ""
        self.__parse(self.__url)

    def __repr__(self):
        """Returns representation of url."""
        return f"URL({self.__url})"

    @property
    def extension(self):
        """Gets the extension of the URL."""
        basename = os.path.basename(self.__path)
        return basename.rsplit(".")[-1] if "." in basename else ""

    @property
    def scheme(self):
        """Gets the scheme of the URL."""
        return self.__scheme

    @property
    def host(self):
        """Gets the host of the URL."""
        return self.__netloc

    @property
    def path(self):
        """Returns the full path to the full."""
        return self.__path

    @property
    def relative_path(self):
        """Returns the relative path to the full."""
        path = self.path
        while path.startswith("/"):
            path = path[1:]
        return path

    def __str__(self):
        """Returns url as string depending on the value of scheme and host."""
        # In this case, it is allowed to violate the guidelines, and to use multiple return statements.
        # As it improves readability of the code
        if self.scheme == "":
            return f"{self.__path}"
        if self.host == "":
            return f"{self.scheme}:{self.__path}"
        return f"{self.scheme}://{self.host}{self.__path}"

    def __hash__(self):
        """Returns hash of url."""
        return hash(str(self))

    def __eq__(self, other):
        """Checks if the url strings are equal."""
        return str(self) == str(other)

    def __len__(self):
        """Returns length of url."""
        return len(self.__url)
