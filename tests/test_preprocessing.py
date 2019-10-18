import sys
import pandas as pd
import unittest
sys.path.append('lib')
import preprocessing

class PreProcessingTest(unittest.TestCase):

    def setUp(self) -> None:
        # Setup pandas dataframe for tests
        data = {
                   'annual_revenue_number':[36844759.88, 5551.79, 264824.80,
                                            896405.29, 829760.26],
                   'annual_revenue_string':['36,844,759.88', '5,551.79', '264,824.80',
                                            '896,405.29', '829,760.26']
                }

        self.df = pd.DataFrame(data)

    def test_logify_feature_and_drop_original_feature(self):
        dataframe = preprocessing.logify_feature(self.df, 'annual_revenue_number')
        result = [17.422224, 8.621876, 12.486824, 13.706148, 13.628892]
        #print(dataframe.annual_revenue_number_log.tolist())
        #print(result)

        df_results = [ round(elem, 2) for elem in dataframe.annual_revenue_number_log.tolist() ]
        result = [ round(elem, 2) for elem in result ]

        self.assertListEqual(result, df_results)

