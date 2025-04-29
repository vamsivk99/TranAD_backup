import argparse

parser = argparse.ArgumentParser(
	description='Time-series anomaly detection using TranAD')
parser.add_argument('--model', metavar='M', type=str, required=True,
					choices=['TranAD', 'LSTM_AD', 'LSTM_Univariate', 'LSTMAutoencoder', 'USAD', 'MTAD_GAT', 'MSCRED', 'CAE_M', 'GDN', 'GTAD', 'TimesNet', 'MERLIN'])
parser.add_argument('--dataset', metavar='D', type=str, required=True,
					choices=['SMAP', 'MSL', 'SMD', 'SWaT', 'WADI', 'UCR', 'NAB', 'Syn'])
parser.add_argument('--entity', metavar='E', type=str, required=False, default=None,
					help='Specific entity within a dataset (e.g., machine-1-1 for SMD)')
parser.add_argument('--retrain', action='store_true', help='retrain model even if it exists')
parser.add_argument('--test', action='store_true', help='test model')
parser.add_argument('--less', action='store_true', help='train using less data')
parser.add_argument('--window', metavar='W', type=int, default=None, help='Specify window size (overrides model default)')
parser.add_argument('--batch', metavar='B', type=int, default=None, help='Specify batch size (overrides model default)')
args = parser.parse_args()