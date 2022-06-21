import sys
import unittest

sys.path.insert(0, "../src/")

from analysis import Analysis

class AnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(inst):
        inst.analysis = Analysis()

    def test_PCA(self):
        self.analysis.runPCA(10)

    @classmethod
    def tearDownClass(inst):
        return

if __name__ == "__main__":
    unittest.main()
