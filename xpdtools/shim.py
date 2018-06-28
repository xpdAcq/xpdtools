"""Shims"""
import numpy as np


class PDFGetterShim:
    """Shim for PDFGetX3"""

    def __init__(self):
        self.config = {"qmax": "testing"}
        self.fq = np.ones(10), np.ones(10)
        self.sq = np.ones(10), np.ones(10)

    def __call__(self, *args, **kwargs):
        print(
            "This is a testing shim for PDFgetx.\n"
            "If you see this message then "
            "you don't have PDFgetx3 installed.\n"
            "The data that comes from this is for testing purposes only "
            "and has no bearing on reality"
        )
        return np.ones(10), np.ones(10)
